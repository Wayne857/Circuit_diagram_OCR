import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

# ====================== 平衡配置参数（核心：兼顾距离优先级+高覆盖率） ======================
TEXT_CONF_THRESH = 0.5
COMP_CONF_THRESH = 0.7
MATCH_SCORE_THRESH = 0.3  # 降低阈值，提升覆盖率
# 去掉距离硬阈值，改为权重软约束

# 类别映射表
CATEGORY_MAP = {"芯片": "chip", "电容": "capacitor", "电阻": "resistor", "接地": "ground"}
INV_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}

# 平衡权重：距离主导，其他特征辅助（兼顾精度和覆盖率）
WEIGHTS = {
    "dist": -0.5,    # 距离权重为主（优先最近）
    "rel_dist": -0.2,# 相对距离辅助
    "orient": 0.15,  # 方位特征保留
    "iou_min": 0.1,  # IOU特征保留
    "size_ratio": 0.05,# 尺寸比辅助
    "text_conf": 0.05, # 置信度辅助
    "comp_conf": 0.05  # 置信度辅助
}

# 可视化颜色配置
COLOR_MAP = {
    "chip": (0, 128, 255), "capacitor": (0, 255, 0), "resistor": (255, 0, 0),
    "ground": (255, 255, 0), "line_connector": (128, 0, 128), "text": (0, 0, 255),
    "match_line": (255, 0, 255)
}

# 特殊字符替换字典
CHAR_REPLACE = {"μ": "u", "Ω": "ohm"}

# ====================== 工具函数（保留原始距离计算，去掉硬阈值） ======================
def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """计算原始像素坐标的中心"""
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

def calc_original_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """计算原始像素欧氏距离"""
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

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

def calculate_match_score(t: Dict, c: Dict) -> float:
    """计算匹配得分：距离主导+多特征辅助（无硬阈值）"""
    t_center = get_center(t["coord"])
    c_center = get_center(c["bbox"])
    t_bbox = t["coord"]
    c_bbox = c["bbox"]
    
    # 核心特征
    d_original = calc_original_distance(t_center, c_center)
    c_w = c_bbox[2] - c_bbox[0]
    c_h = c_bbox[3] - c_bbox[1]
    c_size = max(c_w, c_h) if max(c_w, c_h) > 0 else 1e-6
    d_rel = d_original / c_size
    
    # 辅助特征
    orient_score = calc_relative_orientation_score(t_center, c_center)
    iou_min = calc_iou_min(t_bbox, c_bbox)
    t_area = (t_bbox[2]-t_bbox[0]) * (t_bbox[3]-t_bbox[1])
    c_area = c_w * c_h
    s_ratio = t_area / c_area if c_area > 0 else 0.0
    t_conf = t["conf"]
    c_conf = c["conf"]
    
    # 加权得分（距离主导，无硬过滤）
    score_dist = (1 / (1 + d_original)) * WEIGHTS["dist"]
    score_rel = (1 / (1 + d_rel)) * WEIGHTS["rel_dist"]
    score_orient = orient_score * WEIGHTS["orient"]
    score_iou = iou_min * WEIGHTS["iou_min"]
    score_size = (1 / (1 + s_ratio)) * WEIGHTS["size_ratio"]
    score_t_conf = t_conf * WEIGHTS["text_conf"]
    score_c_conf = c_conf * WEIGHTS["comp_conf"]
    
    total_score = score_dist + score_rel + score_orient + score_iou + score_size + score_t_conf + score_c_conf
    total_score = max(0.0, min(1.0, total_score))
    return total_score

# ====================== 可视化函数（无FreeType依赖） ======================
def replace_special_chars(text: str) -> str:
    for old, new in CHAR_REPLACE.items():
        text = text.replace(old, new)
    return text

def visualize_matching_result(
    matches: List[Dict],
    text_data: List[Dict],
    img_w: int,
    img_h: int,
    save_path: str = "matching_result.png"
) -> None:
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    text_center_map = {}

    # 绘制芯片分割轮廓
    for item in matches:
        comp = item["component"]
        if comp["category"] == "chip" and "segmentation" in comp and comp["segmentation"]:
            seg_points = np.array(comp["segmentation"], dtype=np.int32)
            cv2.polylines(img, [seg_points], isClosed=True, color=COLOR_MAP["chip"], thickness=3)
            comp_center = get_center(comp["bbox"])
            cv2.putText(img, "芯片(chip)", (int(comp_center[0]-30), int(comp_center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAP["chip"], 2)

    # 绘制元件检测框
    for item in matches:
        comp = item["component"]
        c_x1, c_y1, c_x2, c_y2 = comp["bbox"]
        color = COLOR_MAP.get(comp["category"], (100, 100, 100))
        cv2.rectangle(img, (c_x1, c_y1), (c_x2, c_y2), color, 2)
        comp_label = f"{INV_CATEGORY_MAP.get(comp['category'], comp['category'])} {comp['conf']:.2f}"
        cv2.putText(img, comp_label, (c_x1, c_y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 绘制文本
    for text in text_data:
        t_x1, t_y1, t_x2, t_y2 = text["coord"]
        cv2.rectangle(img, (t_x1, t_y1), (t_x2, t_y2), COLOR_MAP["text"], 2)
        display_text = replace_special_chars(text["text"])
        text_center = get_center(text["coord"])
        text_center_map[text["text"]] = text_center
        cv2.putText(img, display_text, (int(text_center[0]-10), int(text_center[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MAP["text"], 1)
        cv2.putText(img, f"{text['conf']:.2f}", (t_x1, t_y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_MAP["text"], 1)

    # 绘制匹配连线
    for item in matches:
        comp = item["component"]
        matched_texts = item["matched_texts"]
        if not matched_texts:
            continue
        comp_center = get_center(comp["bbox"])
        for text in matched_texts:
            text_center = text_center_map.get(text["text"])
            if text_center:
                cv2.line(img, (int(comp_center[0]), int(comp_center[1])), 
                         (int(text_center[0]), int(text_center[1])), 
                         COLOR_MAP["match_line"], 2, lineType=cv2.LINE_AA)

    # 绘制图例
    legend_y = 20
    for cat, color in COLOR_MAP.items():
        if cat == "match_line":
            label = "匹配连线"
        elif cat == "text":
            label = "文本"
        else:
            label = INV_CATEGORY_MAP.get(cat, cat)
        cv2.rectangle(img, (10, legend_y-8), (25, legend_y+8), color, -1)
        cv2.putText(img, label, (30, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        legend_y += 20

    cv2.imwrite(save_path, img)
    print(f"✅ 可视化结果已保存到：{save_path}")
    cv2.imshow("PCB Matching Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ====================== 核心匹配逻辑（平衡版：优先最近+高覆盖率） ======================
def hungarian_matching_for_normal_comps(texts: List[Dict], comps: List[Dict]) -> List[Tuple[int, int]]:
    """普通元件匹配：无硬阈值，得分主导，兼顾覆盖率"""
    t_count = len(texts)
    c_count = len(comps)
    if t_count == 0 or c_count == 0:
        return []
    
    # 构建得分矩阵（无硬过滤，所有类别匹配对都计算得分）
    score_matrix = np.zeros((t_count, c_count))
    for t_idx, t in enumerate(texts):
        for c_idx, c in enumerate(comps):
            score_matrix[t_idx, c_idx] = calculate_match_score(t, c)
    
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
    
    # 步骤4：芯片一对多匹配（分割轮廓包含）
    for chip_idx, chip_comp in enumerate(chip_comps):
        chip_segmentation = chip_comp.get("segmentation", [])
        if not chip_segmentation:
            continue
        for text_idx, text in enumerate(valid_texts):
            if text_idx in matched_text_indices:
                continue
            if CATEGORY_MAP.get(text["category"]) != "chip":
                continue
            text_center = get_center(text["coord"])
            if point_in_polygon(text_center, chip_segmentation):
                for res in match_result:
                    if res["component"]["bbox"] == chip_comp["bbox"]:
                        res["matched_texts"].append(text)
                        break
                matched_text_indices.add(text_idx)
    
    # 步骤5：普通元件匹配（平衡版：无硬阈值+低得分阈值）
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
        cat_matches = hungarian_matching_for_normal_comps(cat_texts, cat_comps)
        for t_idx, c_idx in cat_matches:
            orig_text_idx = normal_text_indices[normal_texts.index(cat_texts[t_idx])]
            comp = cat_comps[c_idx]
            for res in match_result:
                if res["component"]["bbox"] == comp["bbox"]:
                    res["matched_texts"].append(cat_texts[t_idx])
                    break
            matched_text_indices.add(orig_text_idx)
    
    # 步骤6：兜底匹配未归属文本（提升覆盖率）
    for text_idx, text in enumerate(valid_texts):
        if text_idx in matched_text_indices:
            continue
        # 芯片文本归属到芯片
        if CATEGORY_MAP.get(text["category"]) == "chip" and len(chip_comps) > 0:
            for res in match_result:
                if res["component"]["category"] == "chip":
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
                    text_center = get_center(text["coord"])
                    min_dist = float("inf")
                    best_comp = None
                    for c in candidate_comps:
                        c_center = get_center(c["bbox"])
                        dist = calc_original_distance(text_center, c_center)
                        if dist < min_dist:
                            min_dist = dist
                            best_comp = c
                    if best_comp:
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

# ====================== 主函数（使用你的原始数据） ======================
if __name__ == "__main__":
    # 你的文本数据
    text_data = [
        {"text": "TPS3840D", "coord": (107, 161, 213, 176), "category": "芯片", "conf": 0.5773},
        {"text": "10μF", "coord": (16, 222, 53, 240), "category": "电容", "conf": 0.7862},
        {"text": "RESET", "coord": (179, 123, 226, 137), "category": "芯片", "conf": 0.9842},
        {"text": "NC", "coord": (44, 123, 68, 138), "category": "芯片", "conf": 0.982},
        {"text": "GND", "coord": (189, 215, 224, 230), "category": "接地", "conf": 0.9948},
        {"text": "MR", "coord": (99, 119, 127, 137), "category": "芯片", "conf": 0.5773},
        {"text": "VDD", "coord": (142, 91, 176, 104), "category": "芯片", "conf": 0.5773},
        {"text": "100kΩ", "coord": (199, 39, 246, 53), "category": "电阻", "conf": 0.7034},
        {"text": "CT", "coord": (97, 198, 117, 212), "category": "芯片", "conf": 0.6087},
    ]
    
    # 你的元件数据
    component_data = [
        {"category": "capacitor", "bbox": (127, 26, 145, 52), "conf": 0.94, "segmentation": [[135, 25], [135, 30], [135, 31], [135, 31], [134, 32], [127, 32], [127, 40], [134, 40], [135, 41], [135, 41]]},
        {"category": "capacitor", "bbox": (59, 219, 77, 244), "conf": 0.92, "segmentation": [[67, 219], [67, 225], [66, 226], [59, 226], [59, 234], [65, 234], [67, 235], [67, 245], [69, 245], [69, 235]]},
        {"category": "ground", "bbox": (119, 54, 155, 73), "conf": 0.88, "segmentation": [[119, 54], [119, 58], [120, 58], [121, 59], [121, 59], [121, 59], [121, 60], [122, 60], [122, 61], [122, 61]]},
        {"category": "line_connector", "bbox": (255, 126, 265, 136), "conf": 0.88, "segmentation": [[256, 127], [256, 137], [262, 137], [262, 135], [264, 134], [265, 134], [265, 127]]},
        {"category": "ground", "bbox": (50, 262, 87, 280), "conf": 0.87, "segmentation": [[50, 262], [50, 265], [52, 265], [52, 266], [52, 266], [53, 267], [53, 267], [53, 268], [53, 268], [54, 268]]},
        {"category": "resistor", "bbox": (250, 0, 270, 68), "conf": 0.84, "segmentation": [[259, 0], [259, 28], [259, 29], [259, 30], [259, 30], [260, 30], [260, 31], [261, 31], [262, 31], [262, 31]]},
        {"category": "chip", "bbox": (75, 62, 252, 250), "conf": 0.83, "segmentation": [[132, 78], [99, 78], [98, 78], [92, 78], [90, 80], [90, 87], [90, 88], [90, 129], [89, 129], [86, 129], [86, 130], [86, 240], [87, 240], [90, 242], [90, 245], [92, 248], [132, 248], [252, 248], [252, 78], [132, 78]]},
        {"category": "ground", "bbox": (188, 252, 226, 270), "conf": 0.82, "segmentation": [[189, 252], [189, 255], [190, 255], [191, 256], [191, 256], [192, 257], [192, 258], [192, 258], [192, 258], [193, 259]]},
    ]
    
    # 执行匹配
    matches, img_w, img_h = optimal_text_component_matching(text_data, component_data)
    
    # 输出匹配结果
    print("\n===== 最终平衡版匹配结果（优先最近+高覆盖率） =====")
    for idx, item in enumerate(matches):
        comp = item["component"]
        texts = item["matched_texts"]
        comp_info = f"[{idx+1}] 元件：{comp['category']} | 检测框：{comp['bbox']}"
        text_list = [t["text"] for t in texts] if texts else ["无匹配文本"]
        print(f"{comp_info} | 匹配文本：{', '.join(text_list)}")
    
    # 可视化
    visualize_matching_result(matches, text_data, img_w, img_h)