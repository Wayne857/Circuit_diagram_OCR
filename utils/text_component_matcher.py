import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import os
from pathlib import Path
import platform
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ====================== 优化配置参数（核心：提升匹配精度） ======================
TEXT_CONF_THRESH = 0.6  # 提高文本置信度阈值
COMP_CONF_THRESH = 0.75  # 提高元件置信度阈值
MATCH_SCORE_THRESH = 0.5  # 提高匹配得分阈值，减少错误匹配
# 去掉距离硬阈值，改为权重软约束

# 类别映射表
CATEGORY_MAP = {"芯片": "chip", "电容": "capacitor", "电阻": "resistor", "接地": "ground"}
INV_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}

# 优化权重：更强的距离主导，减少错误匹配（提升精度）
WEIGHTS = {
    "dist": -0.7,    # 强化距离权重（优先最近）
    "rel_dist": -0.25,# 相对距离辅助
    "orient": 0.1,   # 方位特征权重降低
    "iou_min": 0.15, # IOU特征权重提升
    "size_ratio": 0.05,# 尺寸比辅助
    "text_conf": 0.1, # 文本置信度权重提升
    "comp_conf": 0.1  # 元件置信度权重提升
}

# 可视化颜色配置
COLOR_MAP = {
    "chip": (0, 128, 255), "capacitor": (0, 255, 0), "resistor": (255, 0, 0),
    "ground": (255, 255, 0), "line_connector": (128, 0, 128), "text": (0, 0, 255),
    "match_line": (255, 0, 255)
}

# 特殊字符替换字典
CHAR_REPLACE = {"μ": "u", "Ω": "ohm"}

# ====================== 新增：跨平台字体配置 ======================
def get_available_font_path():
    """
    获取系统中可用的中文字体路径
    适配Windows/Linux/macOS不同系统
    """
    system = platform.system()
    font_paths = []
    
    # Windows 系统常见中文字体路径
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/arial.ttf"        # 备选英文
        ]
    
    # macOS 系统常见中文字体路径
    elif system == "Darwin":
        font_paths = [
            "/Library/Fonts/SimHei.ttf",        # 黑体
            "/System/Library/Fonts/PingFang.ttc", # 苹方
            "/Library/Fonts/Arial.ttf"          # 备选英文
        ]
    
    # Linux 系统常见中文字体路径
    elif system == "Linux":
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # 开源中文字体
            "/usr/share/fonts/truetype/arphic/ukai.ttc",              # 文鼎楷体
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"         # 备选
        ]
    
    # 检查字体文件是否存在
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    
    # 如果都找不到，返回None使用默认字体
    return None

# 获取全局可用字体路径
AVAILABLE_FONT_PATH = get_available_font_path()


class TextComponentMatcher:
    """
    文本与元件匹配器
    用于将检测到的文本与元件进行匹配关联
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """计算原始像素坐标的中心"""
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2, (y1+y2)/2

    @staticmethod
    def calc_original_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """计算原始像素欧氏距离"""
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    @staticmethod
    def point_in_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
        """射线法判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        for i in range(n):
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[(i+1)%n]
            if min(p1y, p2y) < y <= max(p1y, p2y):
                x_intersect = ((y - p1y) * (p2x - p1x)) / (p2y - p1y) + p1x
                if x <= x_intersect:
                    inside = not inside
        return inside

    @staticmethod
    def calc_relative_orientation_score(t_center: Tuple[float, float], c_center: Tuple[float, float]) -> float:
        """计算方位得分"""
        tx, ty = t_center
        cx, cy = c_center
        dx = tx - cx
        dy = ty - cy
        if dy < -1e-4 and abs(dx) < 1e-4:  # 上
            return 1.0
        elif dx < -1e-4 and abs(dy) < 1e-4:  # 左
            return 1.0
        elif dy < -1e-4 and dx < -1e-4:  # 左上
            return 1.0
        else:  # 其他方位
            return 0.2

    @staticmethod
    def calc_iou_min(t_bbox: Tuple[int, int, int, int], c_bbox: Tuple[int, int, int, int]) -> float:
        """计算IOU变体（原始坐标）"""
        t_x1, t_y1, t_x2, t_y2 = t_bbox
        c_x1, c_y1, c_x2, c_y2 = c_bbox
        inter_x1 = max(t_x1, c_x1)
        inter_y1 = max(t_y1, c_y1)
        inter_x2 = min(t_x2, c_x2)
        inter_y2 = min(t_y2, c_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        min_x1 = min(t_x1, c_x1)
        min_y1 = min(t_y1, c_y1)
        min_x2 = max(t_x2, c_x2)
        min_y2 = max(t_y2, c_y2)
        min_area = (min_x2 - min_x1) * (min_y2 - min_y1)
        return inter_area / min_area if min_area > 0 else 0.0

    def calculate_match_score(self, t: Dict, c: Dict) -> float:
        """计算匹配得分：距离主导+多特征辅助（无硬阈值）"""
        t_center = self.get_center(t["coord"])
        c_center = self.get_center(c["bbox"])
        t_bbox = t["coord"]
        c_bbox = c["bbox"]
        
        # 核心特征
        d_original = self.calc_original_distance(t_center, c_center)
        c_w = c_bbox[2] - c_bbox[0]
        c_h = c_bbox[3] - c_bbox[1]
        c_size = max(c_w, c_h) if max(c_w, c_h) > 0 else 1e-6
        d_rel = d_original / c_size
        
        # 辅助特征
        orient_score = self.calc_relative_orientation_score(t_center, c_center)
        iou_min = self.calc_iou_min(t_bbox, c_bbox)
        t_area = (t_bbox[2]-t_bbox[0]) * (t_bbox[3]-t_bbox[1])
        c_area = c_w * c_h
        s_ratio = t_area / c_area if c_area > 0 else 0.0
        t_conf = t["conf"]
        c_conf = c["conf"]
        
        # 加权得分（距离主导，优先匹配近距离文本）
        # 使用更严格的距离惩罚，近距离文本获得更高得分
        score_dist = (1 / (1 + d_original/50.0)) * WEIGHTS["dist"]  # 除以50使距离影响更显著
        score_rel = (1 / (1 + d_rel)) * WEIGHTS["rel_dist"]
        score_orient = orient_score * WEIGHTS["orient"]
        score_iou = iou_min * WEIGHTS["iou_min"]
        score_size = (1 / (1 + s_ratio)) * WEIGHTS["size_ratio"]
        score_t_conf = t_conf * WEIGHTS["text_conf"]
        score_c_conf = c_conf * WEIGHTS["comp_conf"]
        
        total_score = score_dist + score_rel + score_orient + score_iou + score_size + score_t_conf + score_c_conf
        total_score = max(0.0, min(1.0, total_score))
        return total_score

    @staticmethod
    def replace_special_chars(text: str) -> str:
        for old, new in CHAR_REPLACE.items():
            text = text.replace(old, new)
        return text

    def visualize_matching_result(
        self,
        matches: List[Dict],
        text_data: List[Dict],
        img_w: int,
        img_h: int,
        save_path: str = "matching_result.png"
    ) -> None:
        """可视化匹配结果"""
        # 创建一个更大的画布，为图例预留空间
        legend_width = 150  # 为图例预留宽度
        canvas_w = img_w + legend_width
        canvas = np.ones((img_h, canvas_w, 3), dtype=np.uint8) * 255
        
        text_center_map = {}

        # 在主图区域绘制芯片分割轮廓（偏移到右侧，为图例留出空间）
        for item in matches:
            comp = item["component"]
            if comp["category"] == "chip" and "segmentation" in comp and comp["segmentation"]:
                # 偏移所有坐标，为左侧图例腾出空间
                offset_x = legend_width
                seg_points = np.array(comp["segmentation"], dtype=np.int32)
                seg_points[:, 0] += offset_x  # 只偏移x坐标
                cv2.polylines(canvas, [seg_points], isClosed=True, color=COLOR_MAP["chip"], thickness=3)
                comp_center = self.get_center(comp["bbox"])
                comp_center_with_offset = (comp_center[0] + offset_x, comp_center[1])
                cv2.putText(canvas, "芯片(chip)", (int(comp_center_with_offset[0]-30), int(comp_center_with_offset[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP["chip"], 2)

        # 在主图区域绘制元件检测框（偏移到右侧，为图例留出空间）
        for item in matches:
            comp = item["component"]
            c_x1, c_y1, c_x2, c_y2 = comp["bbox"]
            # 偏移所有坐标，为左侧图例腾出空间
            offset_x = legend_width
            c_x1 += offset_x
            c_x2 += offset_x
            color = COLOR_MAP.get(comp["category"], (100, 100, 100))
            cv2.rectangle(canvas, (c_x1, c_y1), (c_x2, c_y2), color, 2)
            # 使用英文标签确保正常显示
            english_categories = {
                "chip": "Chip",
                "capacitor": "Capacitor",
                "resistor": "Resistor",
                "ground": "Ground",
                "line_connector": "Line Connector"
            }
            category_name = english_categories.get(comp['category'], comp['category'])
            comp_label = f"{category_name} {comp['conf']:.2f}"
            cv2.putText(canvas, comp_label, (c_x1, c_y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 在主图区域绘制文本（偏移到右侧，为图例留出空间）
        for text in text_data:
            t_x1, t_y1, t_x2, t_y2 = text["coord"]
            # 偏移所有坐标，为左侧图例腾出空间
            offset_x = legend_width
            t_x1 += offset_x
            t_x2 += offset_x
            cv2.rectangle(canvas, (t_x1, t_y1), (t_x2, t_y2), COLOR_MAP["text"], 2)
            display_text = self.replace_special_chars(text["text"])
            text_center = self.get_center(text["coord"])
            text_center_with_offset = (text_center[0] + offset_x, text_center[1])
            text_center_map[text["text"]] = text_center_with_offset
            # 使用PIL绘制中文文本
            if PIL_AVAILABLE:
                img_pil = Image.fromarray(canvas)
                draw = ImageDraw.Draw(img_pil)
                try:
                    if AVAILABLE_FONT_PATH:
                        font = ImageFont.truetype(AVAILABLE_FONT_PATH, 12)
                    else:
                        font = ImageFont.load_default()
                except (IOError, Exception):
                    font = ImageFont.load_default()
                draw.text((int(text_center_with_offset[0]-10), int(text_center_with_offset[1])), display_text, font=font, fill=(0, 0, 255))
                canvas = np.array(img_pil)
            else:
                cv2.putText(canvas, display_text, (int(text_center_with_offset[0]-10), int(text_center_with_offset[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MAP["text"], 1)
            cv2.putText(canvas, f"{text['conf']:.2f}", (t_x1, t_y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_MAP["text"], 1)

        # 在主图区域绘制匹配连线（偏移到右侧，为图例留出空间）
        for item in matches:
            comp = item["component"]
            matched_texts = item["matched_texts"]
            if not matched_texts:
                continue
            # 应用偏移量到元件中心
            offset_x = legend_width
            comp_bbox = list(comp["bbox"])
            comp_bbox[0] += offset_x  # x1
            comp_bbox[2] += offset_x  # x2
            comp_center = self.get_center(comp_bbox)
            for text in matched_texts:
                text_center = text_center_map.get(text["text"])
                if text_center:
                    cv2.line(canvas, (int(comp_center[0]), int(comp_center[1])), 
                             (int(text_center[0]), int(text_center[1])), 
                             COLOR_MAP["match_line"], 2, lineType=cv2.LINE_AA)

        # 在最左边绘制图例，上下居中
        total_legend_items = len(COLOR_MAP)
        legend_height_needed = total_legend_items * 20
        # 计算垂直居中的起始y坐标
        start_y = max(20, (img_h - legend_height_needed) // 2)
        
        for idx, (cat, color) in enumerate(COLOR_MAP.items()):
            # 统一使用英文标签
            english_labels = {
                "chip": "Chip",
                "capacitor": "Capacitor",
                "resistor": "Resistor",
                "ground": "Ground",
                "line_connector": "Line Connector",
                "match_line": "Match Line",
                "text": "Text"
            }
            label = english_labels.get(cat, cat)
            
            legend_y = start_y + idx * 20
            
            # 绘制颜色块
            cv2.rectangle(canvas, (10, legend_y-8), (25, legend_y+8), color, -1)
            
            # 使用OpenCV绘制英文标签（确保兼容性）
            cv2.putText(canvas, label, (30, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(save_path, canvas)
        print(f"✅ 可视化结果已保存到： {save_path}")

    def hungarian_matching_for_normal_comps(self, texts: List[Dict], comps: List[Dict]) -> List[Tuple[int, int]]:
        """普通元件匹配：无硬阈值，得分主导，兼顾覆盖率"""
        t_count = len(texts)
        c_count = len(comps)
        if t_count == 0 or c_count == 0:
            return []
        
        # 构建得分矩阵（添加距离硬约束）
        score_matrix = np.zeros((t_count, c_count))
        for t_idx, t in enumerate(texts):
            for c_idx, c in enumerate(comps):
                # 先检查距离约束
                text_center = self.get_center(t["coord"])
                comp_center = self.get_center(c["bbox"])
                dist_to_comp = self.calc_original_distance(text_center, comp_center)
                
                # 计算元件对角线长度作为参考
                comp_diag = self.calc_original_distance(
                    (c["bbox"][0], c["bbox"][1]), 
                    (c["bbox"][2], c["bbox"][3])
                )
                
                # 设置最大允许距离为元件大小加上一定范围
                max_allowed_dist = comp_diag * 1.5  # 调整系数控制匹配范围，但仍允许一定距离的匹配
                
                if dist_to_comp <= max_allowed_dist:
                    score_matrix[t_idx, c_idx] = self.calculate_match_score(t, c)
                else:
                    # 如果超出距离限制，则得分为负数，避免匹配
                    score_matrix[t_idx, c_idx] = -1.0
        
        # 匈牙利算法找最优匹配（低得分阈值提升覆盖率）
        cost_matrix = -score_matrix
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            valid_matches = []
            for t_idx, c_idx in zip(row_ind, col_ind):
                if score_matrix[t_idx, c_idx] > MATCH_SCORE_THRESH:
                    valid_matches.append((t_idx, c_idx))
            return valid_matches
        except ValueError:
            return []

    def optimal_text_component_matching(
        self,
        texts: List[Dict],
        comps: List[Dict]
    ) -> Tuple[List[Dict], int, int]:
        """平衡版主匹配函数"""
        # 自动推断图片尺寸
        all_x = []
        all_y = []
        for t in texts:
            x1, y1, x2, y2 = t["coord"]
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        for c in comps:
            x1, y1, x2, y2 = c["bbox"]
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        img_w = max(all_x) + 10 if all_x else 500
        img_h = max(all_y) + 10 if all_y else 500
        print(f"✅ 自动推断图片尺寸：{img_w}x{img_h}")
        
        # 步骤1：过滤低置信度样本
        valid_texts = [t for t in texts if t["conf"] >= TEXT_CONF_THRESH]
        valid_comps = [c for c in comps if c["conf"] >= COMP_CONF_THRESH]
        
        # 步骤2：分离芯片和普通元件
        chip_comps = [c for c in valid_comps if c["category"] == "chip"]
        normal_comps = [c for c in valid_comps if c["category"] != "chip"]
        
        # 步骤3：初始化匹配结果
        matched_text_indices = set()
        match_result = []
        for c in valid_comps:
            match_result.append({
                "component": c,
                "matched_texts": [],
                "match_type": "chip_contain" if c["category"] == "chip" else "normal_score"
            })
        
        # 步骤4：芯片一对多匹配（分割轮廓包含+距离约束）- 任何在芯片区域内的文本都归属于芯片
        for chip_idx, chip_comp in enumerate(chip_comps):
            chip_segmentation = chip_comp.get("segmentation", [])
            if not chip_segmentation:
                continue
            for text_idx, text in enumerate(valid_texts):
                if text_idx in matched_text_indices:
                    continue
                text_center = self.get_center(text["coord"])
                # 检查文本是否在芯片轮廓内 - 无论文本类别如何，只要在芯片区域内就归属于芯片
                if self.point_in_polygon(text_center, chip_segmentation):
                    # 额外检查距离，确保文本与芯片中心距离合理
                    chip_center = self.get_center(chip_comp["bbox"])
                    dist_to_center = self.calc_original_distance(text_center, chip_center)
                    chip_diag = self.calc_original_distance(
                        (chip_comp["bbox"][0], chip_comp["bbox"][1]), 
                        (chip_comp["bbox"][2], chip_comp["bbox"][3])
                    )
                    max_allowed_dist = chip_diag * 0.8  # 芯片对角线长度的0.8倍
                    if dist_to_center <= max_allowed_dist:
                        for res in match_result:
                            if res["component"]["bbox"] == chip_comp["bbox"]:
                                res["matched_texts"].append(text)
                                break
                        matched_text_indices.add(text_idx)
        
        # 步骤5：普通元件匹配（平衡版：距离约束+得分阈值）
        normal_texts = []
        normal_text_indices = []
        normal_text_cat_map = {}
        for text_idx, text in enumerate(valid_texts):
            if text_idx in matched_text_indices:
                continue
            text_cat = CATEGORY_MAP.get(text["category"])
            if text_cat in [c["category"] for c in normal_comps]:
                normal_texts.append(text)
                normal_text_indices.append(text_idx)
                normal_text_cat_map[len(normal_texts)-1] = text_cat
        
        # 按类别分组匹配（避免跨类别，保证精度）
        for text_cat in set(normal_text_cat_map.values()):
            cat_texts = [normal_texts[i] for i in range(len(normal_texts)) if normal_text_cat_map.get(i) == text_cat]
            cat_comps = [c for c in normal_comps if c["category"] == text_cat]
            cat_matches = self.hungarian_matching_for_normal_comps(cat_texts, cat_comps)
            for t_idx, c_idx in cat_matches:
                orig_text_idx = normal_text_indices[normal_texts.index(cat_texts[t_idx])]
                comp = cat_comps[c_idx]
                
                # 添加距离约束，确保文本和元件距离合理
                text_center = self.get_center(cat_texts[t_idx]["coord"])
                comp_center = self.get_center(comp["bbox"])
                dist_to_comp = self.calc_original_distance(text_center, comp_center)
                
                # 计算元件对角线长度作为参考
                comp_diag = self.calc_original_distance(
                    (comp["bbox"][0], comp["bbox"][1]), 
                    (comp["bbox"][2], comp["bbox"][3])
                )
                
                # 设置最大允许距离为元件大小加上一定范围
                max_allowed_dist = comp_diag * 0.8  # 调整系数控制匹配范围
                
                if dist_to_comp <= max_allowed_dist:
                    for res in match_result:
                        if res["component"]["bbox"] == comp["bbox"]:
                            res["matched_texts"].append(cat_texts[t_idx])
                            break
                    matched_text_indices.add(orig_text_idx)
        
        # 步骤6：兜底匹配未归属文本（提升覆盖率，但遵守距离约束）
        for text_idx, text in enumerate(valid_texts):
            if text_idx in matched_text_indices:
                continue
            # 芯片文本归属到芯片，但要考虑距离因素
            if CATEGORY_MAP.get(text["category"]) == "chip" and len(chip_comps) > 0:
                text_center = self.get_center(text["coord"])
                min_dist = float("inf")
                closest_chip = None
                for chip_comp in chip_comps:
                    chip_center = self.get_center(chip_comp["bbox"])
                    dist = self.calc_original_distance(text_center, chip_center)
                    
                    # 计算芯片对角线长度作为参考
                    chip_diag = self.calc_original_distance(
                        (chip_comp["bbox"][0], chip_comp["bbox"][1]), 
                        (chip_comp["bbox"][2], chip_comp["bbox"][3])
                    )
                    max_allowed_dist = chip_diag * 1.2  # 设置芯片的最大匹配距离
                    
                    if dist < min_dist and dist <= max_allowed_dist:
                        min_dist = dist
                        closest_chip = chip_comp
                
                if closest_chip:
                    for res in match_result:
                        if res["component"]["bbox"] == closest_chip["bbox"]:
                            res["matched_texts"].append(text)
                            matched_text_indices.add(text_idx)
                            break
            # 普通文本归属到同类别最近元件
            else:
                text_cat = CATEGORY_MAP.get(text["category"])
                if text_cat:
                    candidate_comps = [c for c in valid_comps if c["category"] == text_cat]
                    if candidate_comps:
                        # 找最近的元件
                        text_center = self.get_center(text["coord"])
                        min_dist = float("inf")
                        best_comp = None
                        for c in candidate_comps:
                            c_center = self.get_center(c["bbox"])
                            dist = self.calc_original_distance(text_center, c_center)
                            if dist < min_dist:
                                min_dist = dist
                                best_comp = c
                        if best_comp:
                            # 添加距离约束，确保匹配的合理性
                            comp_diag = self.calc_original_distance(
                                (best_comp["bbox"][0], best_comp["bbox"][1]), 
                                (best_comp["bbox"][2], best_comp["bbox"][3])
                            )
                            max_allowed_dist = comp_diag * 1.5  # 设置最大允许距离
                            
                            if min_dist <= max_allowed_dist:
                                for res in match_result:
                                    if res["component"]["bbox"] == best_comp["bbox"]:
                                        res["matched_texts"].append(text)
                                        matched_text_indices.add(text_idx)
                                        break
        
        # 步骤7：补充所有原始元件的结果
        final_result = []
        valid_comp_map = {}
        for item in match_result:
            comp_bbox = tuple(item["component"]["bbox"])
            valid_comp_map[comp_bbox] = item
        
        for c in comps:
            comp_bbox = tuple(c["bbox"])
            if comp_bbox in valid_comp_map:
                final_result.append(valid_comp_map[comp_bbox])
            else:
                final_result.append({
                    "component": c,
                    "matched_texts": [],
                    "match_type": "no_match"
                })
        
        return final_result, img_w, img_h
    
    def format_match_results(self, matches: List[Dict]) -> str:
        """格式化匹配结果，用于写入text_results.txt"""
        result_lines = []
        result_lines.append("\n元件与文本匹配结果:")
        result_lines.append("-" * 50)
        
        for idx, item in enumerate(matches):
            comp = item["component"]
            texts = item["matched_texts"]
            comp_info = f"[{idx+1}] 元件：{comp['category']} | 检测框：{comp['bbox']}"
            text_list = [t["text"] for t in texts] if texts else ["无匹配文本"]
            result_lines.append(f"{comp_info} | 匹配文本：{', '.join(text_list)}")
        
        return "\n".join(result_lines)
    
    def save_visualization(self, matches: List[Dict], text_data: List[Dict], output_dir: str) -> str:
        """保存可视化结果到指定目录"""
        # 自动推断图片尺寸
        all_x = []
        all_y = []
        for t in text_data:
            x1, y1, x2, y2 = t["coord"]
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        for c in matches:
            comp = c["component"]
            x1, y1, x2, y2 = comp["bbox"]
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        img_w = max(all_x) + 10 if all_x else 500
        img_h = max(all_y) + 10 if all_y else 500
        
        # 保存可视化结果
        # 确保输出目录存在
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / "matching_result.png"
        self.visualize_matching_result(matches, text_data, img_w, img_h, str(output_path))
        
        return str(output_path)