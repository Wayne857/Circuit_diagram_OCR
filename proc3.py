import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# -------------------------- 通用工具函数 + 新增引脚专用工具函数 --------------------------
def is_short_segment(cnt, aspect_ratio_thresh=1.5):
    x, y, w, h = cv2.boundingRect(cnt)
    if min(w, h) == 0:
        return False
    aspect_ratio = max(w, h) / min(w, h)
    return aspect_ratio < aspect_ratio_thresh

def explore_four_directions(current_x, current_y, h, w):
    directions = [(current_x + 1, current_y), (current_x - 1, current_y),
                  (current_x, current_y + 1), (current_x, current_y - 1)]
    valid_dirs = []
    for (nx, ny) in directions:
        if 0 <= nx < w and 0 <= ny < h:
            valid_dirs.append((nx, ny))
    return valid_dirs

def get_contour_length(cnt):
    """计算轮廓的实际长度（像素数）"""
    if len(cnt) < 2:
        return 0
    length = 0
    for i in range(1, len(cnt)):
        x1, y1 = cnt[i-1][0]
        x2, y2 = cnt[i][0]
        length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return length

def get_contour_centroid(cnt):
    """计算轮廓质心"""
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return (0, 0)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def is_point_in_neighbor(p, contour_list, thresh=10):
    """判断点p是否在轮廓列表的邻域内（距离<thresh）"""
    px, py = p
    for cnt in contour_list:
        for (x, y) in cnt[:, 0, :]:
            if np.sqrt((x-px)**2 + (y-py)**2) < thresh:
                return True
    return False

def separate_粘连_contour(mask_white, kernel_size=1, dist_kernel=3):
    """分离粘连的白色轮廓（重点分离芯片主体和引脚）"""
    # 距离变换+骨架提取，分离粘连区域
    dist_transform = cv2.distanceTransform(mask_white, cv2.DIST_L2, dist_kernel)
    _, skeleton = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    skeleton = skeleton.astype(np.uint8)
    # 轻微膨胀，恢复引脚宽度
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    separated = cv2.dilate(skeleton, kernel, iterations=1)
    # 提取分离后的轮廓
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return separated, contours

# -------------------------- 核心优化：自适应引脚识别 + 粘连分离 + 二次验证 --------------------------
def calculate_adaptive_pin_thresholds(contours, min_area_percent=0.01, max_area_percent=0.2, 
                                      min_aspect_percent=0.6, min_length_percent=0.02):
    """
    基于图片轮廓统计，计算自适应引脚识别阈值
    :param contours: 所有白色轮廓
    :return: 自适应的pin_min_area/pin_max_area/pin_min_aspect/pin_min_length
    """
    if len(contours) < 5:
        return 5, 100, 3.0, 10  # 轮廓过少时用默认值
    
    # 统计所有轮廓的面积、长宽比、长度
    areas = []
    aspects = []
    lengths = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1:
            continue
        areas.append(area)
        x, y, w, h = cv2.boundingRect(cnt)
        if min(w, h) > 0:
            aspects.append(max(w, h) / min(w, h))
        lengths.append(get_contour_length(cnt))
    
    # 计算分位数（避免极值影响）
    area_99 = np.percentile(areas, 99)
    area_1 = np.percentile(areas, 1)
    aspect_60 = np.percentile(aspects, 60) if len(aspects) > 0 else 3.0
    length_99 = np.percentile(lengths, 99) if len(lengths) > 0 else 50
    
    # 自适应阈值
    pin_min_area = max(area_1, min_area_percent * area_99)
    pin_max_area = min_area_percent * area_99 * 20  # 芯片主体面积远大于引脚
    pin_min_aspect = max(aspect_60, min_aspect_percent * 10)
    pin_min_length = max(min_length_percent * length_99, 8)
    
    return int(pin_min_area), int(pin_max_area), round(pin_min_aspect, 1), int(pin_min_length)

def is_accurate_chip_pin(cnt, chip_body_contours,
                         pin_min_area=5, pin_max_area=100, pin_min_aspect=3.0, pin_min_length=10,
                         pin_angle_tol=8, pin_convexity_thresh=0.8, pin_chip_neighbor_thresh=15):
    """
    高精度芯片引脚识别：多特征融合 + 与芯片主体邻域验证
    :param chip_body_contours: 芯片主体轮廓列表（用于邻域验证）
    :param pin_convexity_thresh: 引脚凸包比（引脚接近直线，凸包比高）
    :param pin_chip_neighbor_thresh: 引脚必须与芯片主体相邻的距离阈值
    :return: (is_pin, score) is_pin=是否为引脚，score=识别置信度[0,1]
    """
    score = 0.0
    area = cv2.contourArea(cnt)
    cnt_length = get_contour_length(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    centroid = get_contour_centroid(cnt)
    
    # 1. 面积筛选（基础）- 置信度+0.2
    if pin_min_area <= area <= pin_max_area:
        score += 0.2
    else:
        return False, 0.0
    
    # 2. 长度筛选（关键：引脚是细长线段）- 置信度+0.2
    if cnt_length >= pin_min_length:
        score += 0.2
    else:
        return False, 0.0
    
    # 3. 长宽比筛选 - 置信度+0.2
    if min(w, h) == 0:
        score += 0.2  # 极细线段直接加分
    else:
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio >= pin_min_aspect:
            score += 0.2
        else:
            return False, 0.0
    
    # 4. 方向筛选：横平竖直（放宽容差到8°，适配轻微倾斜的引脚）- 置信度+0.1
    rect = cv2.minAreaRect(cnt)
    _, _, angle = rect
    if w < h:
        angle = angle - 90
    angle = np.round(angle, 1)
    is_hv = (abs(angle) <= pin_angle_tol) or (abs(angle - 90) <= pin_angle_tol) or (abs(angle + 90) <= pin_angle_tol)
    if is_hv:
        score += 0.1
    
    # 5. 凸包比筛选：引脚接近直线，凸包比高 - 置信度+0.1
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        convexity = area / hull_area
        if convexity >= pin_convexity_thresh:
            score += 0.1
    
    # 6. 核心验证：引脚必须与芯片主体相邻（真正的引脚一定连在芯片上）- 置信度+0.2
    if is_point_in_neighbor(centroid, chip_body_contours, pin_chip_neighbor_thresh):
        score += 0.2
    else:
        return False, score
    
    # 置信度>=0.8才判定为有效引脚
    return score >= 0.8, score

def is_regular_chip_body(cnt, chip_body_min_area=80, chip_aspect_thresh=2.0, chip_compact_thresh=0.45):
    """优化芯片主体识别：放宽阈值，避免漏判小芯片"""
    area = cv2.contourArea(cnt)
    if area < chip_body_min_area:
        return False
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 10.0
    perimeter = cv2.arcLength(cnt, True)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
    if aspect_ratio <= chip_aspect_thresh and compactness >= chip_compact_thresh:
        return True
    return False

def extract_contour_features(cnt, angle_tol=5):
    """提取轮廓特征（用于芯片主体聚类）"""
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    cnt_pts = cnt.reshape(-1, 2)
    pt_num = len(cnt_pts)
    box_area = w * h if (w * h) > 0 else 1.0
    perimeter = cv2.arcLength(cnt, True)

    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 30.0
    aspect_ratio = np.clip(aspect_ratio, 1.0, 30.0) / 30.0
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 1.0
    compactness = np.clip(compactness, 0.0, 1.0)
    rect = cv2.minAreaRect(cnt)
    _, _, angle = rect
    if w < h:
        angle = angle - 90
    angle = np.round(angle, 1)
    is_hv = (abs(angle) <= angle_tol) or (abs(angle - 90) <= angle_tol) or (abs(angle + 90) <= angle_tol)
    dir_feat = 1.0 if is_hv else 0.0
    pix_ratio = area / box_area
    pix_ratio = np.clip(pix_ratio, 0.0, 1.0)
    pt_density = pt_num / (perimeter if perimeter > 0 else 1.0)
    pt_density = np.clip(pt_density, 0.0, 5.0) / 5.0

    return [aspect_ratio, compactness, dir_feat, pix_ratio, pt_density]

def find_and_remove_entire_chip_area(white_seg_mask, chip_body_contours, bg_gray, 
                                     expansion_factor=1.2, max_chip_area_ratio=0.3):
    """
    查找并移除整个芯片区域（包括边界和外轮廓）
    :param white_seg_mask: 白色线段掩码
    :param chip_body_contours: 芯片主体轮廓
    :param bg_gray: 背景灰度
    :param expansion_factor: 区域扩展因子（扩大芯片边界）
    :param max_chip_area_ratio: 芯片最大面积占图像比例
    :return: 完整的芯片区域掩码
    """
    h, w = white_seg_mask.shape
    chip_area_mask = np.zeros((h, w), np.uint8)
    
    if not chip_body_contours:
        return chip_area_mask
    
    # 方法1：如果芯片主体轮廓较大且完整，直接使用其外接矩形并扩展
    for cnt in chip_body_contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # 只处理较大的芯片主体
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            
            # 扩展矩形边界（包含芯片边缘）
            expand_x = int(w_rect * (expansion_factor - 1) / 2)
            expand_y = int(h_rect * (expansion_factor - 1) / 2)
            
            x1 = max(0, x - expand_x)
            y1 = max(0, y - expand_y)
            x2 = min(w, x + w_rect + expand_x)
            y2 = min(h, y + h_rect + expand_y)
            
            # 填充整个矩形区域
            cv2.rectangle(chip_area_mask, (x1, y1), (x2, y2), 255, -1)
    
    return chip_area_mask

def accurate_cluster_remove_chip_body_keep_pins(orig_gray, orig_rgb, bg_gray, white_thresh,
                                                angle_tol=5, min_area=5, dbscan_eps=0.45, dbscan_min_samples=3,
                                                # 引脚精细化可调参数
                                                pin_angle_tol=8, pin_convexity_thresh=0.75, pin_chip_neighbor_thresh=15,
                                                # 芯片主体可调参数
                                                chip_body_min_area=80, chip_aspect_thresh=2.0, chip_compact_thresh=0.45):
    """
    高精度：移除整个芯片区域，只保留芯片引脚（粘连分离+自适应阈值+多特征验证）
    :return: 处理后图、白色线段掩码（含引脚）、芯片区域掩码、引脚掩码、识别统计
    """
    h, w = orig_gray.shape
    # 步骤1：提取白色区域并做粘连分离预处理（核心步骤）
    _, mask_white = cv2.threshold(orig_gray, white_thresh, 255, cv2.THRESH_BINARY)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, np.ones((1,1), np.uint8), iterations=1)
    # 分离粘连的轮廓（芯片主体+引脚）
    mask_white_separated, contours_separated = separate_粘连_contour(mask_white, kernel_size=1)
    total_contours = len(contours_separated)
    if total_contours == 0:
        return orig_gray.copy(), orig_rgb.copy(), np.full((h,w), bg_gray, np.uint8), np.zeros((h,w), np.uint8), np.zeros((h,w), np.uint8), {}

    # 步骤2：先初筛芯片主体（用于后续引脚邻域验证）
    chip_body_contours = []
    other_contours = []
    for cnt in contours_separated:
        if is_regular_chip_body(cnt, chip_body_min_area, chip_aspect_thresh, chip_compact_thresh):
            chip_body_contours.append(cnt)
        else:
            other_contours.append(cnt)
    print(f"🔍 初筛芯片主体：{len(chip_body_contours)}个 | 其他轮廓：{len(other_contours)}个")

    # 步骤3：计算自适应引脚识别阈值
    pin_min_area, pin_max_area, pin_min_aspect, pin_min_length = calculate_adaptive_pin_thresholds(contours_separated)
    print(f"📌 自适应引脚阈值：面积[{pin_min_area},{pin_max_area}] | 最小长宽比{pin_min_aspect} | 最小长度{pin_min_length}px")

    # 步骤4：高精度识别芯片引脚（多特征+邻域验证）
    accurate_pin_contours = []
    pin_scores = []
    non_pin_contours = []
    for cnt in contours_separated:
        is_pin, score = is_accurate_chip_pin(cnt, chip_body_contours,
                                             pin_min_area, pin_max_area, pin_min_aspect, pin_min_length,
                                             pin_angle_tol, pin_convexity_thresh, pin_chip_neighbor_thresh)
        if is_pin:
            accurate_pin_contours.append(cnt)
            pin_scores.append(score)
        else:
            non_pin_contours.append(cnt)
    # 对引脚按置信度排序，保留高置信度
    if len(pin_scores) > 0:
        high_conf_idx = np.where(np.array(pin_scores) >= 0.85)[0]
        high_conf_pins = [accurate_pin_contours[i] for i in high_conf_idx]
    else:
        high_conf_pins = []
    print(f"🔍 引脚识别：初识别{len(accurate_pin_contours)}个 | 高置信保留{len(high_conf_pins)}个（置信度≥0.85）")

    # 步骤5：查找并创建整个芯片区域掩码（关键改进）
    print(f"🔄 查找整个芯片区域...")
    chip_area_mask = find_and_remove_entire_chip_area(mask_white_separated, chip_body_contours, bg_gray)
    
    # 确保引脚不被移除：从芯片区域掩码中减去引脚区域
    pin_mask_only = np.zeros((h, w), np.uint8)
    cv2.drawContours(pin_mask_only, high_conf_pins, -1, 255, -1)
    chip_area_mask = cv2.bitwise_and(chip_area_mask, cv2.bitwise_not(pin_mask_only))
    
    # 对芯片区域掩码进行清理（移除小孔洞）
    chip_area_mask = cv2.morphologyEx(chip_area_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    chip_area_mask = cv2.morphologyEx(chip_area_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    
    # 统计芯片区域面积
    chip_pixel_count = np.sum(chip_area_mask > 0)
    print(f"✅ 芯片区域掩码创建完成：{chip_pixel_count}像素 ({chip_pixel_count/(h*w)*100:.1f}%图像)")

    # 步骤6：生成各类掩码
    pin_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(pin_mask, high_conf_pins, -1, 255, -1)    # 引脚专属掩码（绿色标记）
    
    # 白色线段掩码（包含引脚和其他线段，但要移除芯片区域）
    white_seg_mask = mask_white_separated.copy()
    white_seg_mask = cv2.bitwise_and(white_seg_mask, cv2.bitwise_not(chip_area_mask))
    
    # 步骤7：移除整个芯片区域，只保留引脚
    cluster_removed_gray = orig_gray.copy()
    cluster_removed_rgb = orig_rgb.copy()
    cluster_removed_gray[chip_area_mask == 255] = bg_gray
    cluster_removed_rgb[chip_area_mask == 255] = [bg_gray, bg_gray, bg_gray] if len(orig_rgb.shape)==3 else bg_gray

    # 统计信息
    stats = {
        "total_contours": total_contours,
        "chip_body_num": len(chip_body_contours),
        "pin_num": len(high_conf_pins),
        "chip_area_pixels": chip_pixel_count,
        "chip_area_percent": chip_pixel_count/(h*w)*100,
        "adaptive_pin_thresholds": (pin_min_area, pin_max_area, pin_min_aspect, pin_min_length)
    }
    print(f"📊 最终筛选结果：移除芯片区域{chip_pixel_count}像素 | 保留引脚{len(high_conf_pins)}个")

    return cluster_removed_gray, cluster_removed_rgb, white_seg_mask, chip_area_mask, pin_mask, stats

# -------------------------- 新增：其他元件识别与去除 --------------------------
def remove_other_components(white_seg_mask, pin_mask, chip_body_mask, bg_gray, black_mask=None,
                           min_component_area=100, max_component_area=5000,
                           connectivity_threshold=10):
    """
    识别并去除其他元件（非引脚，非芯片主体的白色区域）
    
    策略：
    1. 排除引脚（已经由pin_mask标识）
    2. 排除芯片主体（已经由chip_body_mask标识）
    3. 对于剩余的白色区域：
       - 如果连接到黑色线段：很可能是线段缺陷，保留
       - 如果不连接到黑色线段：很可能是元件，移除
    """
    h, w = white_seg_mask.shape
    
    print(f"🔍 开始识别其他元件...")
    print(f"  白色掩码像素: {np.sum(white_seg_mask > 0)}")
    print(f"  引脚掩码像素: {np.sum(pin_mask > 0)}")
    print(f"  芯片主体掩码像素: {np.sum(chip_body_mask > 0)}")
    
    # 1. 创建不包括引脚和芯片主体的掩码
    non_pin_chip_mask = white_seg_mask.copy()
    
    # 排除引脚区域
    if np.sum(pin_mask) > 0:
        non_pin_chip_mask = cv2.bitwise_and(non_pin_chip_mask, cv2.bitwise_not(pin_mask))
    
    # 排除芯片主体区域
    if np.sum(chip_body_mask) > 0:
        non_pin_chip_mask = cv2.bitwise_and(non_pin_chip_mask, cv2.bitwise_not(chip_body_mask))
    
    # 2. 提取轮廓
    contours, _ = cv2.findContours(non_pin_chip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"  没有发现其他元件")
        return white_seg_mask.copy(), np.zeros((h, w), np.uint8)
    
    print(f"  发现 {len(contours)} 个候选轮廓")
    
    # 3. 如果有黑色掩码，检查连接性
    black_dilated = None
    if black_mask is not None and np.sum(black_mask) > 0:
        # 膨胀黑色掩码以检测连接性
        black_dilated = cv2.dilate(black_mask, np.ones((connectivity_threshold, connectivity_threshold), np.uint8))
    
    # 4. 识别元件
    component_mask = np.zeros((h, w), np.uint8)
    result_mask = white_seg_mask.copy()
    
    component_count = 0
    defect_count = 0
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # 跳过太小的区域
        if area < min_component_area:
            continue
        
        # 计算轮廓特征
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        
        # 检查是否连接到黑色线段
        connected_to_black = False
        if black_dilated is not None:
            cnt_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            
            # 检查是否有重叠
            overlap = cv2.bitwise_and(cnt_mask, black_dilated)
            if np.sum(overlap) > 0:
                connected_to_black = True
        
        # 判断是否为元件
        # 规则1：面积适中
        # 规则2：不连接到黑色线段（如果是线段缺陷，应该连接到黑色线段）
        is_component = False
        if min_component_area <= area <= max_component_area:
            if not connected_to_black:
                is_component = True
        
        # 处理
        if is_component:
            # 添加到元件掩码
            cv2.drawContours(component_mask, [cnt], -1, 255, -1)
            component_count += 1
            
            # 从结果掩码中移除
            cnt_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            result_mask = cv2.bitwise_and(result_mask, cv2.bitwise_not(cnt_mask))
            
            print(f"  元件 {i+1}: 面积={area:.0f}, 已移除")
        else:
            defect_count += 1
    
    print(f"✅ 元件识别完成: 发现{component_count}个元件, 保留{defect_count}个可能缺陷")
    print(f"  元件掩码像素: {np.sum(component_mask > 0)}")
    
    return result_mask, component_mask


# -------------------------- 黑色线段处理（修复解包错误+鲁棒性优化） --------------------------
def rotate_contour_to_horizontal_vertical(cnt, angle_tol=5):
    rect = cv2.minAreaRect(cnt)
    (x_center, y_center), (w, h), angle = rect
    if w < h:
        angle = angle - 90
    angle = np.round(angle, 1)
    need_correct = False
    target_angle = angle
    if abs(angle) <= angle_tol:
        need_correct = True
        target_angle = 0.0
    elif abs(angle - 90) <= angle_tol or abs(angle + 90) <= angle_tol:
        need_correct = True
        target_angle = 90.0
    if not need_correct:
        return cnt, False
    rotate_angle = target_angle - angle
    M = cv2.getRotationMatrix2D((x_center, y_center), rotate_angle, 1.0)
    cnt_pts = cnt.reshape(-1, 2)
    cnt_pts_rotated = cv2.transform(cnt_pts[np.newaxis, :, :], M)[0]
    cnt_corrected = cnt_pts_rotated.astype(np.int32).reshape(-1, 1, 2)
    return cnt_corrected, True

def normalize_black_segments(mask_black, angle_tol=5, min_area=10):
    h, w = mask_black.shape
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    norm_mask = np.zeros((h, w), np.uint8)
    corrected_count = 0
    norm_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cnt_corrected, is_corrected = rotate_contour_to_horizontal_vertical(cnt, angle_tol)
        if is_corrected:
            corrected_count += 1
        norm_contours.append(cnt_corrected)
        cv2.drawContours(norm_mask, [cnt_corrected], -1, 255, -1)
    norm_mask = cv2.morphologyEx(norm_mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)
    norm_contours, _ = cv2.findContours(norm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"🔍 黑色线段归一化：修正{corrected_count}个非横平竖直轮廓")
    return norm_mask, norm_contours, corrected_count

# 修复：函数名明确，避免与scipy.find_peaks混淆
def find_width_peak(widths, bin_step=0.2, peak_prominence=0.5):
    if len(widths) < 3:
        return np.mean(widths) if widths else 0.0, None, None, 0.0
    min_w, max_w = max(0, np.min(widths)-0.5), np.max(widths)+0.5
    bins = np.arange(min_w, max_w + bin_step, bin_step)
    width_hist, _ = np.histogram(widths, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # 修复：使用重命名后的scipy_find_peaks
    peaks, _ = scipy_find_peaks(width_hist, prominence=peak_prominence)
    if len(peaks) == 0:
        return np.mean(widths), width_hist, bin_centers, np.mean(widths)
    max_peak_idx = np.argmax(bin_centers[peaks])
    max_peak_width = bin_centers[peaks[max_peak_idx]]
    right_valley_width = max_w
    for i in range(peaks[max_peak_idx]+1, len(width_hist)):
        if width_hist[i] < width_hist[i-1] and (i == len(width_hist)-1 or width_hist[i] < width_hist[i+1]):
            right_valley_width = bin_centers[i]
            break
    return max_peak_width, width_hist, bin_centers, right_valley_width

def filter_widths_by_peak(peak_width, right_valley_width, widths, tol_pct=0.2, tol_px=2):
    if peak_width == 0 or len(widths) == 0:
        return widths, np.mean(widths) if widths else 0.0
    filtered = [w for w in widths if peak_width <= w <= right_valley_width]
    if len(filtered) == 0:
        filtered = [w for w in widths if abs(w-peak_width) <= tol_px]
    if len(filtered) == 0:
        filtered = widths
    return filtered, np.mean(filtered)

def unify_black_width(mask_black, target_width, dist_kernel=3):
    if target_width <= 0:
        return mask_black
    dist = cv2.distanceTransform(mask_black, cv2.DIST_L2, dist_kernel)
    max_dist = np.max(dist) if np.max(dist) > 0 else 1
    skeleton = (dist >= max_dist * 0.4).astype(np.uint8) * 255
    expand_r = max(1, int(round(target_width / 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_r*2, expand_r*2))
    unified = cv2.dilate(skeleton, kernel, iterations=1)
    unified = cv2.bitwise_and(unified, mask_black)
    unified = cv2.morphologyEx(unified, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)
    return unified

# -------------------------- 线段修复（兼容高精度引脚掩码） --------------------------
def fit_line_and_get_ends(cnt):
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    cnt_pts = cnt.reshape(-1, 2)
    proj = (cnt_pts[:, 0] - x0) * vx + (cnt_pts[:, 1] - y0) * vy
    p1, p2 = cnt_pts[np.argmin(proj)], cnt_pts[np.argmax(proj)]
    dir_vec = np.array([vx[0], vy[0]])
    return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), dir_vec, -dir_vec

def repair_segments_from_white(mask_black, mask_white, contours_black, target_width, 
                              width_tol_pct=0.2, width_tol_px=2, dist_kernel=3, aspect_ratio_thresh=1.5):
    if target_width <= 0 or np.sum(mask_white) == 0:
        return mask_black
    h, w = mask_black.shape
    mask_repaired = mask_black.copy()
    mask_white_only = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_black))
    visited = np.zeros((h, w), dtype=np.bool_)
    max_step = max(h, w) // 2
    win_size = int(round(target_width * 2))
    half_win = win_size // 2

    for cnt in contours_black:
        if cv2.contourArea(cnt) < 10:
            continue
        is_short = is_short_segment(cnt, aspect_ratio_thresh)
        if not is_short:
            p1, p2, d1, d2 = fit_line_and_get_ends(cnt)
            for ep, dir_vec in [(p1, d1), (p2, d2)]:
                cx, cy = ep
                if not (0<=cx<w and 0<=cy<h):
                    continue
                visited[cy, cx] = True
                for step in range(max_step):
                    nx = int(cx + dir_vec[0] * step)
                    ny = int(cy + dir_vec[1] * step)
                    if not (0<=nx<w and 0<=ny<h) or visited[ny, nx] or mask_white_only[ny, nx] == 0:
                        continue
                    visited[ny, nx] = True
                    y1, y2 = max(0, ny-half_win), min(h, ny+half_win)
                    x1, x2 = max(0, nx-half_win), min(w, nx+half_win)
                    local_white = mask_white_only[y1:y2, x1:x2]
                    if np.sum(local_white) < 3:
                        break
                    dist_local = cv2.distanceTransform(local_white, cv2.DIST_L2, dist_kernel)
                    local_w = 2 * np.mean(dist_local[dist_local>0]) if np.sum(dist_local>0) >0 else 0
                    if abs(local_w-target_width)/target_width > width_tol_pct and abs(local_w-target_width) > width_tol_px:
                        break
                    mask_repaired[y1:y2, x1:x2] = cv2.bitwise_or(mask_repaired[y1:y2, x1:x2], local_white)
        else:
            cnt_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, 1)
            edge_pts = np.argwhere(cnt_mask == 255)
            for y, x in edge_pts:
                if visited[y, x]:
                    continue
                queue = [(x, y)]
                visited[y, x] = True
                while queue:
                    cx, cy = queue.pop(0)
                    for nx, ny in explore_four_directions(cx, cy, h, w):
                        if visited[ny, nx] or mask_white_only[ny, nx] == 0:
                            continue
                        visited[ny, nx] = True
                        y1, y2 = max(0, ny-half_win), min(h, ny+half_win)
                        x1, x2 = max(0, nx-half_win), min(w, nx+half_win)
                        local_white = mask_white_only[y1:y2, x1:x2]
                        if np.sum(local_white) < 3:
                            continue
                        dist_local = cv2.distanceTransform(local_white, cv2.DIST_L2, dist_kernel)
                        local_w = 2 * np.mean(dist_local[dist_local>0]) if np.sum(dist_local>0) >0 else 0
                        if abs(local_w-target_width)/target_width > width_tol_pct and abs(local_w-target_width) > width_tol_px:
                            continue
                        mask_repaired[y1:y2, x1:x2] = cv2.bitwise_or(mask_repaired[y1:y2, x1:x2], local_white)
                        queue.append((nx, ny))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    mask_repaired = cv2.morphologyEx(mask_repaired, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(round(target_width/2)),)*2)
    mask_repaired = cv2.dilate(mask_repaired, dilate_kernel, iterations=1)
    mask_repaired = cv2.bitwise_and(mask_repaired, cv2.bitwise_or(mask_black, mask_white_only))
    return mask_repaired

# -------------------------- 引脚颜色转换（变为线段颜色） --------------------------
def convert_pins_to_line_color(processed_gray, processed_rgb, pin_mask, line_color=0):
    """
    将引脚颜色变为线段颜色
    """
    result_gray = processed_gray.copy()
    result_rgb = processed_rgb.copy()
    
    # 将引脚区域变为线段颜色
    if np.sum(pin_mask) > 0:
        result_gray[pin_mask == 255] = line_color
        
        if len(result_rgb.shape) == 3:
            for c in range(3):
                result_rgb[:, :, c][pin_mask == 255] = line_color
        else:
            result_rgb[pin_mask == 255] = line_color
    
    return result_gray, result_rgb

# -------------------------- 主函数：高精度引脚保留 + 完整流程 --------------------------
def process_pcb_segment_accurate_pin_with_components(image_path,
                        # 引脚高精度可调参数（核心）
                        pin_angle_tol=8, pin_convexity_thresh=0.75, pin_chip_neighbor_thresh=15,
                        # 芯片主体可调参数
                        chip_body_min_area=80, chip_aspect_thresh=2.0, chip_compact_thresh=0.45,
                        # 其他元件去除参数
                        min_component_area=100, max_component_area=5000, connectivity_threshold=15,
                        # 其他基础参数
                        angle_tol=5, peak_prominence=10, valley_prominence=5,  
                        black_min_area=20, black_kernel=(2,2), min_area=5,     
                        dbscan_eps=0.45, dbscan_min_samples=3,
                        width_tol_pct=0.25, width_tol_px=3, dist_kernel=3, width_bin_step=0.2,
                        width_peak_prominence=0.5, aspect_ratio_thresh=1.5,
                        # 引脚颜色
                        line_color=0):
    # 0. 读取原始输入图
    orig_rgb = cv2.imread(image_path)
    if orig_rgb is None:
        raise ValueError(f"图片读取失败，请检查路径：{image_path}")
    orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2GRAY)
    orig_blur = cv2.GaussianBlur(orig_gray, (3, 3), 0.5)
    h, w = orig_gray.shape
    print("="*150)
    print("🚀 PCB高精度处理流程：直方图分析→芯片处理→元件去除→黑色处理→引脚颜色转换→修复")
    print("="*150)

    # 1. 直方图分析（优化低对比度PCB的阈值计算）
    print("\n📊 【步骤1：直方图分析】- 确定背景色+黑白阈值（适配低对比度）")
    hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
    hist_ravel = hist.ravel()
    # 优化：放宽容差，避免背景接近纯白时阈值过窄
    peaks, _ = scipy_find_peaks(hist_ravel, prominence=peak_prominence)
    bg_gray = np.argmax(hist_ravel) if len(peaks)==0 else peaks[np.argmax(hist_ravel[peaks])]
    hist_inv = np.max(hist_ravel) - hist_ravel
    valleys, _ = scipy_find_peaks(hist_inv, prominence=valley_prominence)
    valleys = sorted(valleys)
    
    # 优化：低对比度PCB的阈值调整（背景接近255时）
    if bg_gray > 240:  # 背景接近纯白
        black_thresh = max([v for v in valleys if v < bg_gray], default=bg_gray - 10)
        white_thresh = min([v for v in valleys if v > bg_gray], default=min(bg_gray + 2, 255))
    else:
        black_thresh = max([v for v in valleys if v < bg_gray], default=0)
        white_thresh = min([v for v in valleys if v > bg_gray], default=255)
    
    # 最终阈值修正：确保黑白阈值有足够间隔
    if white_thresh - black_thresh < 5:
        white_thresh = min(black_thresh + 8, 255)
        black_thresh = max(white_thresh - 8, 0)
    
    print(f"✅ 直方图结果：背景灰度={bg_gray} | 黑色阈值=<{black_thresh} | 白色阈值=>{white_thresh}")

    # 2. 核心步骤：高精度移除芯片主体，保留引脚（粘连分离+自适应+多特征验证）
    print("\n⚪ 【步骤2：高精度引脚识别+芯片主体移除】- 粘连分离+自适应阈值+邻域验证")
    cluster_removed_gray, cluster_removed_rgb, white_seg_mask, chip_body_mask, pin_mask, stats = accurate_cluster_remove_chip_body_keep_pins(
        orig_gray=orig_blur, orig_rgb=orig_rgb, bg_gray=bg_gray, white_thresh=white_thresh,
        angle_tol=angle_tol, min_area=min_area, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
        pin_angle_tol=pin_angle_tol, pin_convexity_thresh=pin_convexity_thresh, pin_chip_neighbor_thresh=pin_chip_neighbor_thresh,
        chip_body_min_area=chip_body_min_area, chip_aspect_thresh=chip_aspect_thresh, chip_compact_thresh=chip_compact_thresh
    )
    
    # 生成可视化标记图：芯片主体（红）+ 保留引脚（绿）
    mark_vis_rgb = cluster_removed_rgb.copy()
    # 芯片主体红框
    chip_contours, _ = cv2.findContours(chip_body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mark_vis_rgb, chip_contours, -1, (0,0,255), 2)
    # 引脚绿框（粗框，突出显示）
    pin_contours, _ = cv2.findContours(pin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mark_vis_rgb, pin_contours, -1, (0,255,0), 3)
    print(f"✅ 芯片主体移除完成：共移除{stats.get('chip_area_pixels', 0)}像素，保留引脚{stats.get('pin_num', 0)}个")

    # 3. 初步提取黑色线段（用于元件判断）
    print("\n⚫ 【步骤3：初步提取黑色线段】- 用于元件连接性判断")
    _, mask_black_orig = cv2.threshold(cluster_removed_gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    mask_black_orig = cv2.morphologyEx(mask_black_orig, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, black_kernel))
    # 轻微闭运算，连接细黑线断点
    mask_black_orig = cv2.morphologyEx(mask_black_orig, cv2.MORPH_CLOSE, np.ones((1,1), np.uint8), iterations=1)
    
    # 4. 去除其他元件（新增关键步骤）
    print("\n🔴 【步骤4：去除其他元件】- 基于连接性判断")
    cleaned_white_mask, component_mask = remove_other_components(
        white_seg_mask, pin_mask, chip_body_mask, bg_gray, 
        black_mask=mask_black_orig,
        min_component_area=min_component_area,
        max_component_area=max_component_area,
        connectivity_threshold=connectivity_threshold
    )
    
    # 应用元件移除到图像
    if np.sum(component_mask) > 0:
        cluster_removed_gray[component_mask == 255] = bg_gray
        cluster_removed_rgb[component_mask == 255] = [bg_gray, bg_gray, bg_gray] if len(cluster_removed_rgb.shape)==3 else bg_gray
        print(f"✅ 元件去除完成：{np.sum(component_mask > 0)}像素已设为背景色")
    
    # 5. 引脚颜色转换（变为线段颜色）
    print("\n🔄 【步骤5：引脚颜色转换】- 引脚变为线段颜色")
    pins_converted_gray, pins_converted_rgb = convert_pins_to_line_color(
        cluster_removed_gray, cluster_removed_rgb, pin_mask, line_color
    )
    print(f"✅ 引脚颜色转换完成：{np.sum(pin_mask > 0)}个引脚像素已转为黑色")

    # 6. 黑色线段处理（归一化、宽度统一）
    print("\n⚫ 【步骤6：黑色线段处理】- 归一化+宽度统一（适配细黑线）")
    mask_black_norm, contours_black, corr_cnt = normalize_black_segments(mask_black_orig, angle_tol, black_min_area)
    
    # 宽度统计+统一
    black_widths = []
    long_cnt, short_cnt = 0, 0
    for cnt in contours_black:
        area = cv2.contourArea(cnt)
        if area < black_min_area:
            continue
        cnt_mask = np.zeros((h,w), np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
        dist = cv2.distanceTransform(cnt_mask, cv2.DIST_L2, dist_kernel)
        wd = 2 * np.mean(dist[dist>0]) if np.sum(dist>0) >0 else 0
        if wd > 0:
            black_widths.append(wd)
        if is_short_segment(cnt, aspect_ratio_thresh):
            short_cnt +=1
        else:
            long_cnt +=1
    
    target_width = 0.0
    mask_black_unified = mask_black_norm
    if len(black_widths) > 0:
        peak_w, _, _, rv_w = find_width_peak(black_widths, width_bin_step, width_peak_prominence)
        filtered_w, target_width = filter_widths_by_peak(peak_w, rv_w, black_widths, width_tol_pct, width_tol_px)
        mask_black_unified = unify_black_width(mask_black_norm, target_width, dist_kernel)
        print(f"📏 黑色线段统计：长段{long_cnt} | 短段{short_cnt} | 有效宽度数{len(black_widths)} | 统一宽度{target_width:.2f}px")
    else:
        print(f"📏 黑色线段统计：长段{long_cnt} | 短段{short_cnt} | 无有效宽度，跳过统一")

    # 7. 线段修复（用白色线段修复黑色线段，包括引脚）
    print("\n🔧 【步骤7：线段修复】- 用保留的高精度引脚+白色线段修复黑色（放宽匹配容差）")
    mask_black_repaired = repair_segments_from_white(
        mask_black_unified, cleaned_white_mask, contours_black, target_width,
        width_tol_pct, width_tol_px, dist_kernel, aspect_ratio_thresh
    )
    repair_pix = np.sum(mask_black_repaired == 255) - np.sum(mask_black_unified == 255)
    print(f"✅ 线段修复完成：新增修复像素{repair_pix}个 | 修复后总黑色像素{np.sum(mask_black_repaired == 255)}个")

    # 8. 生成最终图像
    print("\n🎨 【步骤8：生成最终图像】")
    
    # 创建最终图像
    final_gray = pins_converted_gray.copy()
    final_rgb = pins_converted_rgb.copy()
    
    # 将修复后的黑色线段应用到图像上
    final_gray[mask_black_repaired == 255] = line_color
    if len(final_rgb.shape) == 3:
        for c in range(3):
            final_rgb[:, :, c][mask_black_repaired == 255] = line_color
    else:
        final_rgb[mask_black_repaired == 255] = line_color

    # 9. 最终高精度统计
    print(f"\n📊 【最终高精度统计结果】")
    print(f"   → 芯片主体：移除{stats.get('chip_area_pixels', 0)}像素 | 初筛{stats.get('chip_body_num', 0)}个")
    print(f"   → 芯片引脚：保留{stats.get('pin_num', 0)}个 | 已转换为线段颜色")
    print(f"   → 其他元件：移除{np.sum(component_mask > 0)}像素")
    print(f"   → 黑色线段：归一化修正{corr_cnt}个 | 长段{long_cnt} | 短段{short_cnt}")
    print(f"   → 线段修复：修复{repair_pix}像素")
    print("="*150)

    # 可视化：新增引脚专用可视化，2行6列（重点展示引脚识别结果）
    plt.figure(figsize=(48, 20), dpi=120)
    # 第一行：原始图 → 粘连分离后白图 → 芯片主体掩码 → 引脚专属掩码 → 芯片红+引脚绿标记图 → 移除芯片后新图
    plt.subplot(2, 6, 1)
    plt.imshow(cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2RGB))
    plt.title('1. Original Input Image', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 2)
    _, mask_white_show = cv2.threshold(orig_blur, white_thresh, 255, cv2.THRESH_BINARY)
    plt.imshow(mask_white_show, cmap='gray')
    plt.title('2. Original White Areas', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 3)
    plt.imshow(chip_body_mask, cmap='gray')
    plt.title('3. Chip Body Mask (Remove)', fontsize=16, fontweight='bold', color='red', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 4)
    plt.imshow(pin_mask, cmap='gray')
    plt.title(f'4. Chip Pin Mask (Keep: {stats.get("pin_num",0)})', fontsize=16, fontweight='bold', color='green', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 5)
    plt.imshow(cv2.cvtColor(mark_vis_rgb, cv2.COLOR_BGR2RGB))
    plt.title('5. Mark: Chip(Red) + Pin(Green)', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 6)
    plt.imshow(cv2.cvtColor(cluster_removed_rgb, cv2.COLOR_BGR2RGB))
    plt.title('6. After Chip Removal', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')

    # 第二行：元件掩码 → 清理后的白色掩码 → 新图提取黑 → 归一化黑 → 宽度统一黑 → 修复后黑
    plt.subplot(2, 6, 7)
    plt.imshow(component_mask, cmap='gray')
    plt.title(f'7. Component Mask (Removed: {np.sum(component_mask>0)}px)', fontsize=16, fontweight='bold', color='red', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 8)
    plt.imshow(cleaned_white_mask, cmap='gray')
    plt.title('8. Cleaned White Mask', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 9)
    plt.imshow(mask_black_orig, cmap='gray')
    plt.title('9. Black from Processed Img', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 10)
    plt.imshow(mask_black_norm, cmap='gray')
    plt.title(f'10. Normalized Black (Corrected {corr_cnt})', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 11)
    plt.imshow(mask_black_unified, cmap='gray')
    plt.title(f'11. Unified Black (Width: {target_width:.2f}px)', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.subplot(2, 6, 12)
    plt.imshow(cv2.cvtColor(final_rgb, cv2.COLOR_BGR2RGB))
    plt.title('12. Final Result Image', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')

    plt.tight_layout()
    save_path = "pcb_complete_processing.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 高精度结果图保存至：{save_path}")
    # -------------------------- 新增：单独保存第6张图的代码 --------------------------
    # 创建新的绘图窗口，专门绘制第6张图
    plt.figure(figsize=(12, 10), dpi=120)  # 可根据需要调整尺寸
    # 绘制和原第6张图相同的内容
    plt.imshow(cv2.cvtColor(cluster_removed_rgb, cv2.COLOR_BGR2RGB))
    plt.title('6. After Chip Removal', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')  # 关闭坐标轴，和原图保持一致
    plt.tight_layout()
    # 保存单独的第6张图，指定独立的文件名
    single_save_path = "pcb_chip_removal_single.png"
    plt.savefig(single_save_path, dpi=150, bbox_inches='tight')
    plt.show()  # 可选：如果不需要显示单独的图，可注释掉这行
    plt.close()  # 关闭新窗口，释放资源

    # 返回所有关键结果（含高精度引脚掩码）
    return {
        "original_img": {"rgb": orig_rgb, "gray": orig_gray},
        "processed_img": {"rgb": final_rgb, "gray": final_gray},
        "mask": {
            "chip_body": chip_body_mask,
            "pin": pin_mask,
            "white_seg": cleaned_white_mask,
            "component": component_mask,
            "black_orig": mask_black_orig,
            "black_norm": mask_black_norm,
            "black_unified": mask_black_unified,
            "black_repaired": mask_black_repaired
        },
        "vis_img": {
            "mark_rgb": mark_vis_rgb
        },
        "stats": {
            **stats,
            "component_removed_pixels": np.sum(component_mask > 0),
            "black_long_num": long_cnt,
            "black_short_num": short_cnt,
            "black_corrected_num": corr_cnt,
            "black_target_width": target_width,
            "repair_pixel_num": repair_pix,
        },
        "saved_path": save_path
    }


if __name__ == "__main__":
    IMAGE_PATH = r"runs/run15/segmented_out/complex_without_segments.jpg"
    try:
        result = process_pcb_segment_accurate_pin_with_components(
            image_path=IMAGE_PATH,
            # 引脚参数：适配低对比度PCB，放宽限制
            pin_angle_tol=12,
            pin_convexity_thresh=0.65,
            pin_chip_neighbor_thresh=25,
            # 芯片参数
            chip_body_min_area=50,
            # 元件参数
            min_component_area=80,
            max_component_area=3000,
            connectivity_threshold=20,
            # 直方图参数：放宽容差
            peak_prominence=5,
            valley_prominence=2,
            # 引脚颜色
            line_color=0  # 黑色
        )
        print("✅ 处理完成！")
        
    except Exception as e:
        print(f"❌ 执行失败：{str(e)}")
        import traceback
        traceback.print_exc()