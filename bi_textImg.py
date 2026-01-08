import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple

# ====================== 配置参数 ======================
IMG_W, IMG_H = 500, 500  # 电路图尺寸
TEXT_CONF_THRESH = 0.5   # 文本置信度阈值
COMP_CONF_THRESH = 0.7   # 元件置信度阈值
MATCH_SCORE_THRESH = 0.6  # 其他元件匹配得分阈值

# 类别映射表
CATEGORY_MAP = {"芯片": "chip", "电容": "capacitor", "电阻": "resistor", "接地": "ground"}
INV_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}

# 特征权重配置（仅用于电阻/电容/接地）
WEIGHTS = {
    "dist": -0.4,    # 归一化距离
    "rel_dist": -0.2,# 相对距离
    "orient": 0.15,  # 相对方位
    "iou_min": 0.1,  # IOU最小
    "size_ratio": 0.05,# 尺寸比
    "text_conf": 0.05, # 文本置信度
    "comp_conf": 0.05  # 元件置信度
}

# ====================== 核心工具函数（新增点在多边形内的判断） ======================
def normalize_coords(bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """坐标归一化到[0,1]"""
    x1, y1, x2, y2 = bbox
    return x1/img_w, y1/img_h, x2/img_w, y2/img_h

def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """计算原始坐标的中心（非归一化，用于几何判断）"""
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2

def point_in_polygon(point: Tuple[float, float], polygon: List[List[int]]) -> bool:
    """
    射线法判断点是否在多边形内
    :param point: 文本中心坐标 (x,y)
    :param polygon: 芯片分割轮廓坐标 [[x1,y1],[x2,y2]...]
    :return: True=在内部，False=在外部
    """
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        p1x, p1y = polygon[i]
        p2x, p2y = polygon[(i+1)%n]
        # 判断点是否在边的y范围内
        if min(p1y, p2y) < y <= max(p1y, p2y):
            # 计算射线与边的交点x坐标
            x_intersect = ( (y - p1y) * (p2x - p1x) ) / (p2y - p1y) + p1x
            if x <= x_intersect:
                inside = not inside
    return inside

def calc_relative_orientation_score(t_center: Tuple[float, float], c_center: Tuple[float, float]) -> float:
    """计算方位得分（偏好上/左/左上）"""
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

def calc_iou_min(t_bbox: Tuple[float, float, float, float], c_bbox: Tuple[float, float, float, float]) -> float:
    """计算IOU变体（最小包围盒）"""
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

def calculate_match_score(t: Dict, c: Dict, img_w: int, img_h: int) -> float:
    """计算文本-元件匹配得分（仅用于电阻/电容/接地）"""
    t_bbox_norm = normalize_coords(t["coord"], img_w, img_h)
    c_bbox_norm = normalize_coords(c["bbox"], img_w, img_h)
    t_cx, t_cy = get_center(t["coord"])  # 用原始中心坐标
    c_cx, c_cy = get_center(c["bbox"])
    t_w, t_h = t_bbox_norm[2]-t_bbox_norm[0], t_bbox_norm[3]-t_bbox_norm[1]
    c_w, c_h = c_bbox_norm[2]-c_bbox_norm[0], c_bbox_norm[3]-c_bbox_norm[1]
    c_size = max(c_w, c_h) if max(c_w, c_h) > 0 else 1e-6
    
    d_norm = np.sqrt((t_cx - c_cx)**2 + (t_cy - c_cy)**2)
    d_rel = d_norm / c_size
    orient_score = calc_relative_orientation_score((t_cx, t_cy), (c_cx, c_cy))
    iou_min = calc_iou_min(t_bbox_norm, c_bbox_norm)
    t_area = t_w * t_h
    c_area = c_w * c_h
    s_ratio = t_area / c_area if c_area > 0 else 0.0
    t_conf = t["conf"]
    c_conf = c["conf"]
    
    score_dist = (1 / (1 + d_norm)) * WEIGHTS["dist"]
    score_rel = (1 / (1 + d_rel)) * WEIGHTS["rel_dist"]
    score_orient = orient_score * WEIGHTS["orient"]
    score_iou = iou_min * WEIGHTS["iou_min"]
    score_size = (1 / (1 + s_ratio)) * WEIGHTS["size_ratio"]
    score_t_conf = t_conf * WEIGHTS["text_conf"]
    score_c_conf = c_conf * WEIGHTS["comp_conf"]
    
    total_score = score_dist + score_rel + score_orient + score_iou + score_size + score_t_conf + score_c_conf
    denominator = sum(WEIGHTS.values()) - WEIGHTS["dist"] - WEIGHTS["rel_dist"]
    denominator = denominator if denominator != 0 else 1e-6
    total_score = (total_score - WEIGHTS["dist"] - WEIGHTS["rel_dist"]) / denominator
    return max(0.0, min(1.0, total_score))

# ====================== 核心匹配逻辑（芯片基于分割轮廓一对多匹配） ======================
def hungarian_matching_for_normal_comps(texts: List[Dict], comps: List[Dict], score_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """匈牙利算法（仅用于电阻/电容/接地的一对一匹配）"""
    if score_matrix.size == 0 or len(texts) == 0 or len(comps) == 0:
        return []
    cost_matrix = -score_matrix
    try:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid_matches = []
        for t_idx, c_idx in zip(row_ind, col_ind):
            if 0 <= t_idx < len(texts) and 0 <= c_idx < len(comps):
                if score_matrix[t_idx, c_idx] > MATCH_SCORE_THRESH:
                    valid_matches.append((t_idx, c_idx))
        return valid_matches
    except ValueError:
        return []

def optimal_text_component_matching(
    texts: List[Dict],
    comps: List[Dict],
    img_w: int,
    img_h: int
) -> List[Dict]:
    """
    最终匹配函数：
    1. 芯片：基于分割轮廓包含关系，一对多匹配所有芯片类文本
    2. 其他元件：加权得分+匈牙利算法一对一匹配
    """
    # 步骤1：过滤有效文本和元件
    valid_texts = [t for t in texts if t["conf"] >= TEXT_CONF_THRESH]
    valid_comps = [c for c in comps if c["conf"] >= COMP_CONF_THRESH]
    
    # 步骤2：分离芯片元件和普通元件（电阻/电容/接地/line_connector）
    chip_comps = [c for c in valid_comps if c["category"] == "chip"]
    normal_comps = [c for c in valid_comps if c["category"] != "chip"]
    
    # 步骤3：芯片优先匹配（基于分割轮廓包含关系，一对多）
    # 记录已匹配的文本索引，避免重复匹配
    matched_text_indices = set()
    # 初始化所有元件的匹配结果
    match_result = []
    for c in valid_comps:
        match_result.append({
            "component": c,
            "matched_texts": [],
            "match_type": "chip_contain" if c["category"] == "chip" else "normal_score"
        })
    
    # 芯片匹配逻辑：遍历每个芯片，匹配所有分类为芯片且中心在分割轮廓内的文本
    for chip_idx, chip_comp in enumerate(chip_comps):
        # 芯片的分割轮廓坐标（从元件数据中取segmentation字段）
        chip_segmentation = chip_comp.get("segmentation", [])
        if not chip_segmentation:
            continue
        # 遍历所有有效文本
        for text_idx, text in enumerate(valid_texts):
            if text_idx in matched_text_indices:
                continue
            # 条件1：文本分类为芯片
            if CATEGORY_MAP.get(text["category"]) != "chip":
                continue
            # 条件2：文本中心在芯片分割轮廓内
            text_center = get_center(text["coord"])
            if point_in_polygon(text_center, chip_segmentation):
                # 添加到芯片的匹配列表
                for res in match_result:
                    if res["component"]["bbox"] == chip_comp["bbox"]:
                        res["matched_texts"].append(text)
                        break
                matched_text_indices.add(text_idx)
    
    # 步骤4：普通元件匹配（电阻/电容/接地，一对一）
    # 筛选未匹配的文本 + 分类与普通元件匹配的文本
    normal_texts = []
    normal_text_indices = []
    for text_idx, text in enumerate(valid_texts):
        if text_idx in matched_text_indices:
            continue
        text_cat = CATEGORY_MAP.get(text["category"])
        if text_cat in [c["category"] for c in normal_comps]:
            normal_texts.append(text)
            normal_text_indices.append(text_idx)
    
    # 构建普通元件的得分矩阵
    t_count = len(normal_texts)
    c_count = len(normal_comps)
    score_matrix = np.zeros((t_count, c_count))
    for t_idx, text in enumerate(normal_texts):
        text_cat = CATEGORY_MAP.get(text["category"])
        for c_idx, comp in enumerate(normal_comps):
            if comp["category"] == text_cat:
                score = calculate_match_score(text, comp, img_w, img_h)
                score_matrix[t_idx, c_idx] = score
    
    # 匈牙利算法匹配
    normal_matches = hungarian_matching_for_normal_comps(normal_texts, normal_comps, score_matrix)
    for t_idx, c_idx in normal_matches:
        text = normal_texts[t_idx]
        comp = normal_comps[c_idx]
        # 添加到普通元件的匹配列表
        for res in match_result:
            if res["component"]["bbox"] == comp["bbox"]:
                res["matched_texts"].append(text)
                break
        matched_text_indices.add(normal_text_indices[t_idx])
    
    # 步骤5：兜底匹配（未匹配的芯片类文本，全部归属到芯片）
    for text_idx, text in enumerate(valid_texts):
        if text_idx in matched_text_indices:
            continue
        if CATEGORY_MAP.get(text["category"]) == "chip" and len(chip_comps) > 0:
            # 归属到第一个芯片（可根据距离调整归属到最近的芯片）
            for res in match_result:
                if res["component"]["category"] == "chip":
                    res["matched_texts"].append(text)
                    matched_text_indices.add(text_idx)
                    break
    
    # 步骤6：补充所有原始元件的结果
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
    
    return final_result

if __name__ == "__main__":
    # 你的原始文本数据
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
    
    # 你的原始元件数据（补充segmentation字段，即芯片的分割轮廓坐标）
    component_data = [
        {"category": "capacitor", "bbox": (127, 26, 145, 52), "conf": 0.94, "segmentation": []},
        {"category": "capacitor", "bbox": (59, 219, 77, 244), "conf": 0.92, "segmentation": []},
        {"category": "ground", "bbox": (119, 54, 155, 73), "conf": 0.88, "segmentation": []},
        {"category": "line_connector", "bbox": (255, 126, 265, 136), "conf": 0.88, "segmentation": []},
        {"category": "ground", "bbox": (50, 262, 87, 280), "conf": 0.87, "segmentation": []},
        {"category": "resistor", "bbox": (250, 0, 270, 68), "conf": 0.84, "segmentation": []},
        {"category": "chip", "bbox": (75, 62, 252, 250), "conf": 0.83, 
         "segmentation": [[132, 78], [99, 78], [98, 78], [92, 78], [90, 80], [90, 87], [90, 88], [90, 129], [89, 129], [86, 129], [86, 130], [86, 240], [87, 240], [90, 242], [90, 245], [92, 248], [132, 248], [252, 248], [252, 78], [132, 78]]},  # 补充完整的芯片分割轮廓
        {"category": "ground", "bbox": (188, 252, 226, 270), "conf": 0.82, "segmentation": []},
    ]
    
    # 执行匹配
    matches = optimal_text_component_matching(text_data, component_data, IMG_W, IMG_H)
    
    # 格式化输出结果
    print("===== 最终匹配结果（芯片一对多+分割轮廓判断） =====")
    for idx, item in enumerate(matches):
        comp = item["component"]
        texts = item["matched_texts"]
        comp_info = f"[{idx+1}] 元件：{comp['category']} | 检测框：{comp['bbox']} | 匹配方式：{item['match_type']}"
        text_list = [t["text"] for t in texts] if texts else ["无匹配文本"]
        text_info = f"匹配文本：{', '.join(text_list)}"
        print(f"{comp_info} | {text_info}")