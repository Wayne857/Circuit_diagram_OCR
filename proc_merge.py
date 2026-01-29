import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.ndimage import gaussian_filter1d, binary_erosion, generate_binary_structure
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict
import os
import random
from scipy.spatial.distance import cdist

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
    dist_transform = cv2.distanceTransform(mask_white, cv2.DIST_L2, dist_kernel)
    _, skeleton = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    skeleton = skeleton.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    separated = cv2.dilate(skeleton, kernel, iterations=1)
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return separated, contours

# -------------------------- 核心优化：自适应引脚识别 + 粘连分离 + 二次验证 --------------------------
def calculate_adaptive_pin_thresholds(contours, min_area_percent=0.01, max_area_percent=0.2,
                                      min_aspect_percent=0.6, min_length_percent=0.02):
    if len(contours) < 5:
        return 5, 100, 3.0, 10
    
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
    
    area_99 = np.percentile(areas, 99)
    area_1 = np.percentile(areas, 1)
    aspect_60 = np.percentile(aspects, 60) if len(aspects) > 0 else 3.0
    length_99 = np.percentile(lengths, 99) if len(lengths) > 0 else 50
    
    pin_min_area = max(area_1, min_area_percent * area_99)
    pin_max_area = min_area_percent * area_99 * 20
    pin_min_aspect = max(aspect_60, min_aspect_percent * 10)
    pin_min_length = max(min_length_percent * length_99, 8)
    return int(pin_min_area), int(pin_max_area), round(pin_min_aspect, 1), int(pin_min_length)

def is_accurate_chip_pin(cnt, chip_body_contours,
                         pin_min_area=5, pin_max_area=100, pin_min_aspect=3.0, pin_min_length=10,
                         pin_angle_tol=8, pin_convexity_thresh=0.8, pin_chip_neighbor_thresh=15):
    score = 0.0
    area = cv2.contourArea(cnt)
    cnt_length = get_contour_length(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    centroid = get_contour_centroid(cnt)
    
    if pin_min_area <= area <= pin_max_area:
        score += 0.2
    else:
        return False, 0.0
    
    if cnt_length >= pin_min_length:
        score += 0.2
    else:
        return False, 0.0
    
    if min(w, h) == 0:
        score += 0.2
    else:
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio >= pin_min_aspect:
            score += 0.2
        else:
            return False, 0.0
    
    rect = cv2.minAreaRect(cnt)
    _, _, angle = rect
    if w < h:
        angle = angle - 90
    angle = np.round(angle, 1)
    is_hv = (abs(angle) <= pin_angle_tol) or (abs(angle - 90) <= pin_angle_tol) or (abs(angle + 90) <= pin_angle_tol)
    if is_hv:
        score += 0.1
    
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        convexity = area / hull_area
        if convexity >= pin_convexity_thresh:
            score += 0.1
    
    if is_point_in_neighbor(centroid, chip_body_contours, pin_chip_neighbor_thresh):
        score += 0.2
    else:
        return False, score
    
    return score >= 0.8, score

def is_regular_chip_body(cnt, chip_body_min_area=80, chip_aspect_thresh=2.0, chip_compact_thresh=0.45):
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
    h, w = white_seg_mask.shape
    chip_area_mask = np.zeros((h, w), np.uint8)
    if not chip_body_contours:
        return chip_area_mask
    
    for cnt in chip_body_contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w_rect, h_rect = cv2.boundingRect(cnt)
            expand_x = int(w_rect * (expansion_factor - 1) / 2)
            expand_y = int(h_rect * (expansion_factor - 1) / 2)
            x1 = max(0, x - expand_x)
            y1 = max(0, y - expand_y)
            x2 = min(w, x + w_rect + expand_x)
            y2 = min(h, y + h_rect + expand_y)
            cv2.rectangle(chip_area_mask, (x1, y1), (x2, y2), 255, -1)
    return chip_area_mask

def accurate_cluster_remove_chip_body_keep_pins(orig_gray, orig_rgb, bg_gray, white_thresh,
                                                angle_tol=5, min_area=5, dbscan_eps=0.45, dbscan_min_samples=3,
                                                pin_angle_tol=8, pin_convexity_thresh=0.75, pin_chip_neighbor_thresh=15,
                                                chip_body_min_area=80, chip_aspect_thresh=2.0, chip_compact_thresh=0.45):
    h, w = orig_gray.shape
    _, mask_white = cv2.threshold(orig_gray, white_thresh, 255, cv2.THRESH_BINARY)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, np.ones((1,1), np.uint8), iterations=1)
    mask_white_separated, contours_separated = separate_粘连_contour(mask_white, kernel_size=1)
    total_contours = len(contours_separated)
    if total_contours == 0:
        return orig_gray.copy(), orig_rgb.copy(), np.full((h,w), bg_gray, np.uint8), np.zeros((h,w), np.uint8), np.zeros((h,w), np.uint8), {}
    
    chip_body_contours = []
    other_contours = []
    for cnt in contours_separated:
        if is_regular_chip_body(cnt, chip_body_min_area, chip_aspect_thresh, chip_compact_thresh):
            chip_body_contours.append(cnt)
        else:
            other_contours.append(cnt)
    print(f"🔍 初筛芯片主体：{len(chip_body_contours)}个 | 其他轮廓：{len(other_contours)}个")
    
    pin_min_area, pin_max_area, pin_min_aspect, pin_min_length = calculate_adaptive_pin_thresholds(contours_separated)
    print(f"📌 自适应引脚阈值：面积[{pin_min_area},{pin_max_area}] | 最小长宽比{pin_min_aspect} | 最小长度{pin_min_length}px")
    
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
    
    if len(pin_scores) > 0:
        high_conf_idx = np.where(np.array(pin_scores) >= 0.85)[0]
        high_conf_pins = [accurate_pin_contours[i] for i in high_conf_idx]
    else:
        high_conf_pins = []
    print(f"🔍 引脚识别：初识别{len(accurate_pin_contours)}个 | 高置信保留{len(high_conf_pins)}个（置信度≥0.85）")
    
    print(f"🔄 查找整个芯片区域...")
    chip_area_mask = find_and_remove_entire_chip_area(mask_white_separated, chip_body_contours, bg_gray)
    pin_mask_only = np.zeros((h, w), np.uint8)
    cv2.drawContours(pin_mask_only, high_conf_pins, -1, 255, -1)
    chip_area_mask = cv2.bitwise_and(chip_area_mask, cv2.bitwise_not(pin_mask_only))
    chip_area_mask = cv2.morphologyEx(chip_area_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    chip_area_mask = cv2.morphologyEx(chip_area_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8), iterations=1)
    chip_pixel_count = np.sum(chip_area_mask > 0)
    print(f"✅ 芯片区域掩码创建完成：{chip_pixel_count}像素 ({chip_pixel_count/(h*w)*100:.1f}%图像)")
    
    pin_mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(pin_mask, high_conf_pins, -1, 255, -1)
    white_seg_mask = mask_white_separated.copy()
    white_seg_mask = cv2.bitwise_and(white_seg_mask, cv2.bitwise_not(chip_area_mask))
    
    cluster_removed_gray = orig_gray.copy()
    cluster_removed_rgb = orig_rgb.copy()
    cluster_removed_gray[chip_area_mask == 255] = bg_gray
    cluster_removed_rgb[chip_area_mask == 255] = [bg_gray, bg_gray, bg_gray] if len(orig_rgb.shape)==3 else bg_gray
    
    stats = {
        "total_contours": total_contours,
        "chip_body_num": len(chip_body_contours),
        "pin_num": len(high_conf_pins),
        "chip_area_pixels": chip_pixel_count,
        "chip_area_percent": chip_pixel_count/(h*w)*100,
        "adaptive_pin_thresholds": (pin_min_area, pin_max_area, pin_min_aspect, pin_min_length)
    }
    print(f"📊 最终筛选结果：移除芯片区域{chip_pixel_count}像素 | 保留引脚{len(high_conf_pins)}个")
    # 修复：返回 chip_area_mask（内部变量名）
    return cluster_removed_gray, cluster_removed_rgb, white_seg_mask, chip_area_mask, pin_mask, stats

# -------------------------- 【核心改进】网格扫描法精确检测线宽 + 元件移除（背景色填充+线段变黑） --------------------------
def detect_components_from_image(img_rgb, bg_gray, white_threshold, black_threshold=None,
                                 min_area_ratio=0.0005, num_grid_samples=20, max_width_ratio=0.1, line_color=0):
    """
    从图像中检测并移除非线段元件（白色区域）：
    - 使用20×20网格扫描精确检测线宽
    - 符合线宽特征的区域 → 变为黑色（线段颜色）
    - 不符合线宽特征的区域 → 设为背景色（彻底移除）
    - 新增：在将矩形框内白色区域变成黑色时，会检查矩形两端是否有黑色线段，如果有则需要延长矩形框以便连接
    """
    print(f"🔍 【元件检测模块】开始分析非芯片白色区域（网格扫描线宽验证 + 背景色移除/线段变黑）...")
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) if len(img_rgb.shape) == 3 else img_rgb.copy()
    h, w = img_gray.shape
    total_pixels = h * w
    
    # 如果未提供黑色阈值，则使用一个经验值
    if black_threshold is None:
        black_threshold = max(0, bg_gray - 30)
    
    # 1. 提取白色区域
    _, white_mask = cv2.threshold(img_gray, white_threshold, 255, cv2.THRESH_BINARY)
    kernel_size = max(1, min(h, w) // 200)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # 2. 提取黑色区域（用于检测矩形框两端的黑色线段）
    _, black_mask = cv2.threshold(img_gray, black_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 轮廓检测
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(20, int(total_pixels * min_area_ratio))
    regions_info = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(cnt)
        regions_info.append({
            'id': i + 1,
            'area': area,
            'bbox': (x, y, width, height),
            'aspect_ratio': max(width, height) / max(min(width, height), 1),
        })
    
    # 4. DBSCAN聚类白色像素
    white_coords = np.column_stack(np.where(white_mask > 0))
    clustered_mask = np.zeros_like(white_mask)
    clustered_regions_raw = []
    
    if len(white_coords) > 0:
        clustering = DBSCAN(eps=10, min_samples=5)
        cluster_labels = clustering.fit_predict(white_coords)
        
        for idx, coord in enumerate(white_coords):
            row, col = coord
            cluster_id = cluster_labels[idx]
            if cluster_id != -1:
                clustered_mask[row, col] = cluster_id + 1
        
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        for cluster_id in unique_clusters:
            cluster_coords = white_coords[cluster_labels == cluster_id]
            y_coords = cluster_coords[:, 0]
            x_coords = cluster_coords[:, 1]
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            width = max_x - min_x
            height = max_y - min_y
            area = len(cluster_coords)
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            
            clustered_regions_raw.append({
                'id': cluster_id + 1,
                'area': area,
                'bbox': (min_x, min_y, width, height),
                'aspect_ratio': aspect_ratio,
            })
        
        # 过滤接近整图尺寸的区域
        filtered_clustered_regions = []
        for region in clustered_regions_raw:
            x, y, width, height = region['bbox']
            width_ratio = width / w
            height_ratio = height / h
            if width_ratio > 0.8 or height_ratio > 0.8:
                continue
            filtered_clustered_regions.append(region)
        clustered_regions = filtered_clustered_regions
    else:
        clustered_regions = []
    
    # 5. 【核心改进】网格扫描法精确检测线宽
    print(f"📏 【线宽检测】使用{num_grid_samples}×{num_grid_samples}网格扫描法...")
    max_width_threshold = int(min(h, w) * max_width_ratio)
    candidate_widths = []
    
    # 水平扫描：20条水平线
    y_coords = [int(i * h / (num_grid_samples + 1)) for i in range(1, num_grid_samples + 1)]
    for y in y_coords:
        x = 0
        while x < w:
            if img_gray[y, x] <= bg_gray:  # 黑色像素
                start = x
                while x < w and img_gray[y, x] <= bg_gray:
                    x += 1
                segment_length = x - start
                if 1 < segment_length < max_width_threshold:
                    candidate_widths.append(segment_length)
            else:
                x += 1
    
    # 垂直扫描：20条垂直线
    x_coords = [int(i * w / (num_grid_samples + 1)) for i in range(1, num_grid_samples + 1)]
    for x in x_coords:
        y = 0
        while y < h:
            if img_gray[y, x] <= bg_gray:  # 黑色像素
                start = y
                while y < h and img_gray[y, x] <= bg_gray:
                    y += 1
                segment_length = y - start
                if 1 < segment_length < max_width_threshold:
                    candidate_widths.append(segment_length)
            else:
                y += 1
    
    # K-means聚类主导线宽
    avg_dominant_width = 0.0
    if len(candidate_widths) > 0:
        X = np.array(candidate_widths).reshape(-1, 1)
        n_clusters = min(len(set(candidate_widths)), 5)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster_idx = unique_labels[np.argmax(counts)]
            dominant_widths = X[labels == dominant_cluster_idx].flatten()
            avg_dominant_width = np.mean(dominant_widths)
        else:
            avg_dominant_width = np.mean(candidate_widths)
    
    print(f"  → 检测到 {len(candidate_widths)} 个有效线宽样本 | 主导线宽: {avg_dominant_width:.2f}px")
    
    # 6. 【关键修改】元件移除 + 线段变黑：
    #    - 不符合线宽特征 → 设为背景色（彻底移除）
    #    - 符合线宽特征 → 变为黑色（线段颜色）
    #    - 新增：在将矩形框内白色区域变成黑色时，会检查矩形两端是否有黑色线段，如果有则需要延长矩形框以便连接
    component_mask = np.zeros_like(white_mask)
    line_segment_mask = np.zeros_like(white_mask)
    result_img_rgb = img_rgb.copy()
    result_img_gray = img_gray.copy()
    
    def extend_rectangle_to_connect_black(bbox, black_mask, max_extension=20):
        """
        检查矩形框两端是否有黑色线段，如果有则延长矩形框以便连接
        参数:
            bbox: (x, y, width, height)
            black_mask: 黑色区域掩码
            max_extension: 最大延长像素数
        返回:
            延长后的bbox: (new_x, new_y, new_width, new_height)
        """
        x, y, width, height = bbox
        
        # 判断矩形框方向（水平或垂直）
        is_horizontal = width > height
        
        if is_horizontal:
            # 水平矩形框：检查左右两端
            # 获取矩形框中部的y坐标
            mid_y = y + height // 2
            
            # 向左探索
            left_extend = 0
            for ext in range(1, max_extension + 1):
                check_x = x - ext
                if check_x < 0:
                    break
                # 检查黑色掩码在矩形中部附近是否有黑色像素
                check_range_y_start = max(0, mid_y - 2)
                check_range_y_end = min(h, mid_y + 3)
                if np.any(black_mask[check_range_y_start:check_range_y_end, check_x] > 0):
                    left_extend = ext
                    # 继续探索直到没有黑色像素
                    for next_ext in range(ext + 1, max_extension + 1):
                        next_x = x - next_ext
                        if next_x < 0:
                            break
                        if not np.any(black_mask[check_range_y_start:check_range_y_end, next_x] > 0):
                            break
                        left_extend = next_ext
        
            # 向右探索
            right_extend = 0
            for ext in range(1, max_extension + 1):
                check_x = x + width - 1 + ext
                if check_x >= w:
                    break
                # 检查黑色掩码在矩形中部附近是否有黑色像素
                check_range_y_start = max(0, mid_y - 2)
                check_range_y_end = min(h, mid_y + 3)
                if np.any(black_mask[check_range_y_start:check_range_y_end, check_x] > 0):
                    right_extend = ext
                    # 继续探索直到没有黑色像素
                    for next_ext in range(ext + 1, max_extension + 1):
                        next_x = x + width - 1 + next_ext
                        if next_x >= w:
                            break
                        if not np.any(black_mask[check_range_y_start:check_range_y_end, next_x] > 0):
                            break
                        right_extend = next_ext
        
            # 更新矩形框
            new_x = x - left_extend
            new_width = width + left_extend + right_extend
            return (new_x, y, new_width, height)
        
        else:
            # 垂直矩形框：检查上下两端
            # 获取矩形框中部的x坐标
            mid_x = x + width // 2
            
            # 向上探索
            top_extend = 0
            for ext in range(1, max_extension + 1):
                check_y = y - ext
                if check_y < 0:
                    break
                # 检查黑色掩码在矩形中部附近是否有黑色像素
                check_range_x_start = max(0, mid_x - 2)
                check_range_x_end = min(w, mid_x + 3)
                if np.any(black_mask[check_y, check_range_x_start:check_range_x_end] > 0):
                    top_extend = ext
                    # 继续探索直到没有黑色像素
                    for next_ext in range(ext + 1, max_extension + 1):
                        next_y = y - next_ext
                        if next_y < 0:
                            break
                        if not np.any(black_mask[next_y, check_range_x_start:check_range_x_end] > 0):
                            break
                        top_extend = next_ext
        
            # 向下探索
            bottom_extend = 0
            for ext in range(1, max_extension + 1):
                check_y = y + height - 1 + ext
                if check_y >= h:
                    break
                # 检查黑色掩码在矩形中部附近是否有黑色像素
                check_range_x_start = max(0, mid_x - 2)
                check_range_x_end = min(w, mid_x + 3)
                if np.any(black_mask[check_y, check_range_x_start:check_range_x_end] > 0):
                    bottom_extend = ext
                    # 继续探索直到没有黑色像素
                    for next_ext in range(ext + 1, max_extension + 1):
                        next_y = y + height - 1 + next_ext
                        if next_y >= h:
                            break
                        if not np.any(black_mask[next_y, check_range_x_start:check_range_x_end] > 0):
                            break
                        bottom_extend = next_ext
        
            # 更新矩形框
            new_y = y - top_extend
            new_height = height + top_extend + bottom_extend
            return (x, new_y, width, new_height)
    
    for region in clustered_regions:
        x, y, width, height = region['bbox']
        min_dimension = min(width, height)
        tolerance = avg_dominant_width * 2.0 if avg_dominant_width > 0 else 10
        
        # 判断是否为线段（符合线宽特征）
        is_line_segment = False
        if avg_dominant_width > 0:
            if abs(min_dimension - avg_dominant_width) <= tolerance:
                is_line_segment = True
        
        # 创建区域掩码
        roi_mask = np.zeros_like(white_mask)
        
        if is_line_segment:
            # 符合线宽：变为黑色（线段颜色）
            # 首先检查是否需要延长矩形框以连接黑色线段
            extended_bbox = extend_rectangle_to_connect_black((x, y, width, height), black_mask, max_extension=20)
            x_ext, y_ext, width_ext, height_ext = extended_bbox
            
            # 使用延长后的矩形框
            cv2.rectangle(roi_mask, (x_ext, y_ext), (x_ext + width_ext, y_ext + height_ext), 255, -1)
            roi_white = cv2.bitwise_and(white_mask, roi_mask)
            
            line_segment_mask = cv2.bitwise_or(line_segment_mask, roi_white)
            result_img_gray[y_ext:y_ext+height_ext, x_ext:x_ext+width_ext][roi_white[y_ext:y_ext+height_ext, x_ext:x_ext+width_ext] > 0] = line_color
            result_img_rgb[y_ext:y_ext+height_ext, x_ext:x_ext+width_ext][roi_white[y_ext:y_ext+height_ext, x_ext:x_ext+width_ext] > 0] = [line_color, line_color, line_color]
        else:
            # 不符合线宽：设为背景色（彻底移除）
            cv2.rectangle(roi_mask, (x, y), (x + width, y + height), 255, -1)
            roi_white = cv2.bitwise_and(white_mask, roi_mask)
            
            component_mask = cv2.bitwise_or(component_mask, roi_white)
            result_img_gray[y:y+height, x:x+width][roi_white[y:y+height, x:x+width] > 0] = bg_gray
            result_img_rgb[y:y+height, x:x+width][roi_white[y:y+height, x:x+width] > 0] = [bg_gray, bg_gray, bg_gray]
    
    # 7. 可视化：创建带红/绿框标记的验证图（仅用于可视化）
    validated_img = img_rgb.copy()
    for region in clustered_regions:
        x, y, width, height = region['bbox']
        min_dimension = min(width, height)
        tolerance = avg_dominant_width * 2.0 if avg_dominant_width > 0 else 10
        is_line_segment = False
        if avg_dominant_width > 0:
            if abs(min_dimension - avg_dominant_width) <= tolerance:
                is_line_segment = True
        
        # 用红色框标记元件区域，绿色框标记线段区域
        if is_line_segment:
            # 对于线段，使用延长后的矩形框进行可视化
            extended_bbox = extend_rectangle_to_connect_black((x, y, width, height), black_mask, max_extension=20)
            x_ext, y_ext, width_ext, height_ext = extended_bbox
            cv2.rectangle(validated_img, (x_ext, y_ext), (x_ext + width_ext, y_ext + height_ext), (0, 255, 0), 2)
            cv2.putText(validated_img, f'{width}x{height} (LINE)', (x_ext, y_ext - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.rectangle(validated_img, (x, y), (x + width, y + height), (0, 0, 255), 2)
            cv2.putText(validated_img, f'{width}x{height} (COMP)', (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 统计
    component_pixels = np.sum(component_mask > 0)
    line_segment_pixels = np.sum(line_segment_mask > 0)
    print(f"✅ 元件检测完成：识别出 {len([r for r in clustered_regions if not (avg_dominant_width > 0 and abs(min(r['bbox'][2], r['bbox'][3]) - avg_dominant_width) <= avg_dominant_width * 2.0)])} 个元件 | "
          f"已移除 {component_pixels} 像素（设为背景色{bg_gray}）| "
          f"线段 {line_segment_pixels} 像素（变为黑色{line_color}）")
    
    return {
        'component_mask': component_mask,
        'line_segment_mask': line_segment_mask,
        'white_mask': white_mask,
        'clustered_mask': clustered_mask,
        'clustered_regions': clustered_regions,
        'line_widths': candidate_widths,
        'avg_dominant_width': avg_dominant_width,
        'bg_gray_level': bg_gray,
        'white_threshold': white_threshold,
        'black_threshold': black_threshold,
        'validated_img': validated_img,  # 可视化用：红框=元件，绿框=线段
        'result_img_rgb': result_img_rgb,  # 实际处理结果
        'result_img_gray': result_img_gray,
        'component_pixels': component_pixels,
        'line_segment_pixels': line_segment_pixels,
        'scan_coords': (y_coords, x_coords)
    }

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

def find_width_peak(widths, bin_step=0.2, peak_prominence=0.5):
    if len(widths) < 3:
        return np.mean(widths) if widths else 0.0, None, None, 0.0
    min_w, max_w = max(0, np.min(widths)-0.5), np.max(widths)+0.5
    bins = np.arange(min_w, max_w + bin_step, bin_step)
    width_hist, _ = np.histogram(widths, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
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
    visited = np.zeros((h, w), dtype=bool)  # 修复：np.bool_ → bool
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
    result_gray = processed_gray.copy()
    result_rgb = processed_rgb.copy()
    if np.sum(pin_mask) > 0:
        result_gray[pin_mask == 255] = line_color
        if len(result_rgb.shape) == 3:
            for c in range(3):
                result_rgb[:, :, c][pin_mask == 255] = line_color
        else:
            result_rgb[pin_mask == 255] = line_color
    return result_gray, result_rgb

# -------------------------- L型线段多边形拟合函数 --------------------------
def fit_polygon_to_L_shape(cnt, epsilon_factor=0.02):
    """
    使用多边形拟合L形状
    参数:
        cnt: L形状轮廓
        epsilon_factor: 多边形拟合的精度因子
    返回:
        polygon: 拟合后的多边形点
        is_valid: 是否有效的L形状多边形
    """
    # 计算轮廓周长
    perimeter = cv2.arcLength(cnt, True)
    
    # 使用多边形近似
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # 简化多边形：如果顶点数过多，进一步简化
    if len(approx) > 8:
        epsilon = 0.05 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # 确保多边形有4-6个顶点（L形状的特征）
    if 4 <= len(approx) <= 8:
        return approx, True
    else:
        # 如果顶点数不合适，返回凸包
        hull = cv2.convexHull(cnt)
        return hull, False

def detect_L_shape_contour(cnt, L_shape_min_area=200):
    """
    检测是否为L形状轮廓
    返回: True表示是L形状，False表示不是
    """
    area = cv2.contourArea(cnt)
    if area < L_shape_min_area:
        return False
    
    # 计算轮廓的多边形近似
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    approx_vertices = len(approx)
    
    # 计算凸包缺陷
    hull = cv2.convexHull(cnt, returnPoints=False)
    defect_count = 0
    if len(hull) > 3:
        defects = cv2.convexityDefects(cnt, hull)
        defect_count = 0 if defects is None else len(defects)
    
    # 计算面积比
    rect = cv2.minAreaRect(cnt)
    (rx, ry), (rw, rh), rangle = rect
    rect_area = rw * rh
    area_ratio = area / rect_area if rect_area > 0 else 1.0
    
    # 计算长宽比
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = max(w, h) / max(min(w, h), 1)
    
    # L形检测条件
    L_shape_conditions = (
        area_ratio < 0.85 and  # 面积比小于0.85表示有缺口
        approx_vertices >= 4 and approx_vertices <= 8 and  # 顶点数在4-8之间
        defect_count >= 1 and  # 有凸包缺陷
        min(rw, rh) > 5 and  # 最小尺寸足够大
        aspect_ratio < 5  # 不是特别细长的形状
    )
    
    return L_shape_conditions

def process_L_shape_with_polygon_fitting(cnt, line_color=0):
    """
    使用多边形拟合L形状并填充黑色
    参数:
        cnt: L形状轮廓
        line_color: 线段颜色（黑色）
    返回:
        polygon: 拟合后的多边形
        area: 多边形面积
        bbox: 边界框
    """
    # 拟合多边形
    polygon, is_valid = fit_polygon_to_L_shape(cnt)
    
    # 计算边界框
    x, y, w, h = cv2.boundingRect(polygon)
    area = cv2.contourArea(polygon)
    
    return polygon, area, (x, y, w, h), is_valid

# -------------------------- 新增：合并相近矩形函数 --------------------------
def merge_close_rectangles(rect_info, avg_line_width, merge_distance_factor=2.0):
    """
    合并距离相近的矩形（在2个平均线段宽度内）
    
    参数:
        rect_info: 矩形信息列表
        avg_line_width: 平均线段宽度
        merge_distance_factor: 合并距离因子（乘以平均线段宽度）
    
    返回:
        merged_rect_info: 合并后的矩形信息列表
        merge_pairs: 合并对信息
    """
    if len(rect_info) <= 1:
        return rect_info, []
    
    # 计算合并距离阈值
    merge_distance = avg_line_width * merge_distance_factor
    
    # 准备数据：计算每个矩形的中心点和边界
    rect_centers = []
    rect_bboxes = []
    
    for rect in rect_info:
        x, y, w, h = rect['bbox']
        center_x = x + w / 2
        center_y = y + h / 2
        rect_centers.append((center_x, center_y))
        rect_bboxes.append((x, y, x + w, y + h))
    
    # 构建距离矩阵
    n_rects = len(rect_info)
    distance_matrix = np.zeros((n_rects, n_rects))
    
    for i in range(n_rects):
        for j in range(i+1, n_rects):
            # 计算中心点距离
            center_i = np.array(rect_centers[i])
            center_j = np.array(rect_centers[j])
            center_distance = np.linalg.norm(center_i - center_j)
            
            # 计算边界框之间的最小距离
            bbox_i = rect_bboxes[i]
            bbox_j = rect_bboxes[j]
            
            # 计算两个矩形边界之间的距离
            # 如果矩形有重叠，距离为0
            x_overlap = max(0, min(bbox_i[2], bbox_j[2]) - max(bbox_i[0], bbox_j[0]))
            y_overlap = max(0, min(bbox_i[3], bbox_j[3]) - max(bbox_i[1], bbox_j[1]))
            
            if x_overlap > 0 and y_overlap > 0:
                # 矩形重叠，距离为0
                bbox_distance = 0
            else:
                # 计算最近边界距离
                dx = max(bbox_i[0] - bbox_j[2], bbox_j[0] - bbox_i[2], 0)
                dy = max(bbox_i[1] - bbox_j[3], bbox_j[1] - bbox_i[3], 0)
                bbox_distance = np.sqrt(dx*dx + dy*dy)
            
            # 使用边界框距离作为主要判断标准
            distance_matrix[i, j] = bbox_distance
            distance_matrix[j, i] = bbox_distance
    
    # 查找需要合并的矩形对
    merge_pairs = []
    merged_indices = set()
    
    for i in range(n_rects):
        if i in merged_indices:
            continue
        
        # 查找与当前矩形距离小于阈值的其他矩形
        close_rects = []
        for j in range(i+1, n_rects):
            if j in merged_indices:
                continue
            
            if distance_matrix[i, j] < merge_distance:
                close_rects.append(j)
        
        if close_rects:
            # 合并当前矩形和所有相近矩形
            merge_group = [i] + close_rects
            merge_pairs.append(merge_group)
            
            # 标记已合并的矩形
            merged_indices.update(merge_group)
    
    # 创建合并后的矩形列表
    merged_rect_info = []
    merged_rect_count = 0
    
    # 首先添加未合并的矩形
    for i, rect in enumerate(rect_info):
        if i not in merged_indices:
            # 创建新ID
            new_rect = rect.copy()
            new_rect['id'] = len(merged_rect_info) + 1
            new_rect['is_merged'] = False
            merged_rect_info.append(new_rect)
    
    # 合并矩形组
    for group_idx, group in enumerate(merge_pairs):
        if len(group) == 0:
            continue
        
        # 计算合并后的边界框
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for rect_idx in group:
            x, y, w, h = rect_info[rect_idx]['bbox']
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
        
        new_w = max_x - min_x
        new_h = max_y - min_y
        new_area = new_w * new_h
        new_aspect_ratio = max(new_w, new_h) / max(min(new_w, new_h), 1)
        
        # 选择一种颜色（使用第一个矩形的颜色）
        group_color = rect_info[group[0]]['color']
        
        # 创建合并后的矩形 - 修复：添加缺失的键
        merged_rect = {
            'id': len(merged_rect_info) + 1,
            'bbox': (int(min_x), int(min_y), int(new_w), int(new_h)),
            'area': new_area,
            'aspect_ratio': new_aspect_ratio,
            'color': group_color,
            'is_split': False,
            'is_merged': True,
            'is_polygon': False,  # 合并的是矩形，不是多边形
            'parent_id': None,
            'shape_type': 'merged_rectangle',
            'merged_from': [rect_info[idx]['id'] for idx in group],
            'merge_group_size': len(group)
        }
        
        merged_rect_info.append(merged_rect)
        merged_rect_count += 1
    
    print(f"   → 合并了 {len(merge_pairs)} 组矩形，共合并 {sum(len(group) for group in merge_pairs)} 个矩形为 {merged_rect_count} 个新矩形")
    
    return merged_rect_info, merge_pairs

def draw_rectangles_and_polygons_on_black_segments(final_rgb, mask_black_repaired, min_area=10, thickness=2,
                                                   detect_L_shapes=True, L_shape_min_area=200,
                                                   merge_close_rects=True, avg_line_width=0, line_color=0):
    """
    在最终图像上绘制黑色线段的矩形逼近和多边形拟合
    对L型线段使用多边形拟合并填充黑色
    
    参数:
        final_rgb: 原始RGB图像
        mask_black_repaired: 修复后的黑色线段掩码
        min_area: 最小面积阈值
        thickness: 矩形线宽
        detect_L_shapes: 是否启用L形检测
        L_shape_min_area: L形检测的最小面积
        merge_close_rects: 是否合并相近矩形
        avg_line_width: 平均线段宽度（用于合并判断）
        line_color: 线段颜色（用于填充）
    
    返回:
        final_with_annotations: 带有矩形和多边形标注的图像
        shape_info: 矩形和多边形信息列表
        mask_updated: 更新后的黑色掩码
    """
    final_with_annotations = final_rgb.copy()
    mask_updated = mask_black_repaired.copy()
    
    # 从黑色掩码中提取轮廓
    contours, _ = cv2.findContours(mask_black_repaired, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_info = []  # 存储形状信息（包括矩形和多边形）
    
    # 生成一系列鲜艳的颜色
    colors = [
        (255, 0, 0),    # 蓝色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 黄色
        (255, 128, 0),  # 橙色
        (128, 255, 0),  # 浅绿
        (0, 128, 255),  # 浅蓝
        (255, 0, 128),  # 粉红
        (128, 0, 255),  # 紫红
        (0, 255, 128),  # 青绿
        (255, 128, 128),# 浅粉
        (128, 255, 128),# 浅青绿
        (128, 128, 255),# 浅蓝紫
        (255, 255, 128),# 浅黄
        (255, 128, 255),# 浅紫
        (128, 255, 255),# 浅青
    ]
    
    print(f"🔍 处理黑色线段轮廓：共{len(contours)}个轮廓")
    
    # 首先处理所有L形状
    L_shape_count = 0
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 检测是否为L形状
        if detect_L_shapes and area >= L_shape_min_area:
            is_L_shape = detect_L_shape_contour(cnt, L_shape_min_area)
            
            if is_L_shape:
                L_shape_count += 1
                print(f"   → 检测到L形状轮廓 #{i+1}: 面积={area:.1f}")
                
                # 使用多边形拟合L形状
                polygon, polygon_area, bbox, is_valid = process_L_shape_with_polygon_fitting(cnt, line_color)
                x, y, w, h = bbox
                
                # 为多边形分配颜色
                color_idx = len(shape_info) % len(colors)
                color = colors[color_idx]
                
                # 保存多边形信息
                shape_info.append({
                    'id': len(shape_info) + 1,
                    'bbox': (x, y, w, h),
                    'area': polygon_area,
                    'aspect_ratio': max(w, h) / max(min(w, h), 1),
                    'color': color,
                    'shape_type': 'L_shape_polygon',
                    'polygon': polygon,  # 存储多边形点
                    'is_polygon': True,
                    'is_valid_fit': is_valid
                })
                
                # 在多边形掩码中填充黑色
                cv2.drawContours(mask_updated, [polygon], -1, 255, -1)
                
                # 在图像上绘制多边形边界
                cv2.polylines(final_with_annotations, [polygon], True, color, thickness)
                
                # 添加标签
                label_text = f"#{shape_info[-1]['id']}: L"
                font_scale = max(0.3, min(0.6, 20 / max(w, h)))
                cv2.putText(final_with_annotations, label_text, 
                           (x, max(0, y-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                
                # 在原始图像中填充多边形区域为黑色
                if len(final_with_annotations.shape) == 3:
                    # 创建多边形掩码
                    polygon_mask = np.zeros_like(mask_black_repaired)
                    cv2.drawContours(polygon_mask, [polygon], -1, 255, -1)
                    # 填充黑色
                    final_with_annotations[polygon_mask == 255] = [line_color, line_color, line_color]
                else:
                    polygon_mask = np.zeros_like(mask_black_repaired)
                    cv2.drawContours(polygon_mask, [polygon], -1, 255, -1)
                    final_with_annotations[polygon_mask == 255] = line_color
    
    print(f"   → 共检测到 {L_shape_count} 个L形状，已进行多边形拟合")
    
    # 处理非L形状的轮廓（矩形）
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        
        # 跳过已经处理过的L形状
        # 这里我们需要检查当前轮廓是否已经被处理为L形状
        # 由于我们之前已经处理了L形状，现在只处理非L形状
        if detect_L_shapes and area >= L_shape_min_area:
            is_L_shape = detect_L_shape_contour(cnt, L_shape_min_area)
            if is_L_shape:
                continue  # 跳过已经处理过的L形状
        
        # 为非L形状分配颜色
        color_idx = len(shape_info) % len(colors)
        color = colors[color_idx]
        
        # 保存矩形信息
        shape_info.append({
            'id': len(shape_info) + 1,
            'bbox': (x, y, w, h),
            'area': area,
            'aspect_ratio': aspect_ratio,
            'color': color,
            'shape_type': 'rectangle' if aspect_ratio > 1.5 else 'square',
            'is_polygon': False,
            'is_merged': False
        })
        
        # 绘制矩形边界
        cv2.rectangle(final_with_annotations, (x, y), (x + w, y + h), color, thickness)
        
        # 添加标签
        label_text = f"#{shape_info[-1]['id']}: {w}x{h}"
        font_scale = max(0.3, min(0.6, 20 / max(w, h)))
        cv2.putText(final_with_annotations, label_text, (x, max(0, y-5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
    # ========== 合并相近矩形（仅对非多边形形状） ==========
    if merge_close_rects and avg_line_width > 0 and len(shape_info) > 0:
        print(f"\n🔗 【矩形合并】开始合并距离相近的矩形（阈值: {2*avg_line_width:.1f}px）...")
        
        # 分离多边形和矩形
        polygon_shapes = [s for s in shape_info if s.get('is_polygon', False)]
        rectangle_shapes = [s for s in shape_info if not s.get('is_polygon', True)]  # 使用get方法，默认值为True
        
        if len(rectangle_shapes) > 1:
            # 只对矩形进行合并
            merged_rect_info, merge_pairs = merge_close_rectangles(rectangle_shapes, avg_line_width)
            
            if len(merge_pairs) > 0:
                print(f"   → 重新绘制合并后的矩形...")
                # 重新绘制图像，显示合并后的矩形
                final_with_annotations = final_rgb.copy()
                
                # 首先绘制多边形
                for shape in polygon_shapes:
                    if shape.get('is_polygon', False):  # 使用get方法
                        polygon = shape.get('polygon')
                        if polygon is not None:
                            color = shape['color']
                            x, y, w, h = shape['bbox']
                            
                            # 绘制多边形边界
                            cv2.polylines(final_with_annotations, [polygon], True, color, thickness)
                            
                            # 添加标签
                            label_text = f"#{shape['id']}: L"
                            font_scale = max(0.3, min(0.6, 20 / max(w, h)))
                            cv2.putText(final_with_annotations, label_text, 
                                       (x, max(0, y-5)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                            
                            # 填充多边形区域为黑色
                            polygon_mask = np.zeros_like(mask_black_repaired)
                            cv2.drawContours(polygon_mask, [polygon], -1, 255, -1)
                            if len(final_with_annotations.shape) == 3:
                                final_with_annotations[polygon_mask == 255] = [line_color, line_color, line_color]
                            else:
                                final_with_annotations[polygon_mask == 255] = line_color
                
                # 然后绘制合并后的矩形
                for shape in merged_rect_info:
                    x, y, w, h = shape['bbox']
                    color = shape['color']
                    thickness = 2
                    
                    # 绘制矩形
                    cv2.rectangle(final_with_annotations, (x, y), (x + w, y + h), color, thickness)
                    
                    # 添加标签
                    if shape.get('is_merged', False):
                        label_text = f"M#{shape['id']}: {w}x{h}"
                    else:
                        label_text = f"#{shape['id']}: {w}x{h}"
                    
                    font_scale = max(0.3, min(0.6, 20 / max(w, h)))
                    cv2.putText(final_with_annotations, label_text, (x, max(0, y-5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                    
                    # 如果是合并后的矩形，在掩码中填充黑色
                    if shape.get('is_merged', False):
                        cv2.rectangle(mask_updated, (x, y), (x + w, y + h), 255, -1)
                        # 在图像中填充黑色
                        if len(final_with_annotations.shape) == 3:
                            final_with_annotations[y:y+h, x:x+w] = [line_color, line_color, line_color]
                        else:
                            final_with_annotations[y:y+h, x:x+w] = line_color
                
                # 更新形状信息
                shape_info = polygon_shapes + merged_rect_info
                print(f"   → 矩形合并完成，现在有 {len(shape_info)} 个形状（{len(polygon_shapes)}个多边形 + {len(merged_rect_info)}个矩形）")
            else:
                print(f"   → 没有需要合并的矩形")
        else:
            print(f"   → 矩形数量不足，跳过合并")
    else:
        print(f"\n🔗 【矩形合并】跳过合并步骤（条件不满足）")
    
    # 统计信息 - 修复：使用get方法避免KeyError
    polygon_count = sum(1 for s in shape_info if s.get('is_polygon', False))
    rectangle_count = sum(1 for s in shape_info if not s.get('is_polygon', True))
    merged_count = sum(1 for s in shape_info if s.get('is_merged', False))
    
    print(f"📐 形状标注完成：共{len(shape_info)}个形状（{polygon_count}个多边形 + {rectangle_count}个矩形）")
    
    if polygon_count > 0:
        print(f"   → L形状多边形拟合：{polygon_count}个L形状已进行多边形拟合并填充黑色")
    
    if merged_count > 0:
        print(f"   → 矩形合并：{merged_count}个矩形是通过合并相近矩形得到的")
    
    return final_with_annotations, shape_info, mask_updated

# -------------------------- 主函数：高精度引脚保留 + 完整流程（集成网格扫描+背景色移除+线段变黑） --------------------------
def process_pcb_segment_accurate_pin_with_components(image_path,
    pin_angle_tol=8, pin_convexity_thresh=0.75, pin_chip_neighbor_thresh=15,
    chip_body_min_area=80, chip_aspect_thresh=2.0, chip_compact_thresh=0.45,
    min_area_ratio=0.0005, num_grid_samples=20,
    angle_tol=5, black_min_area=20, black_kernel=(2,2), min_area=5,
    dbscan_eps=0.45, dbscan_min_samples=3,
    width_tol_pct=0.25, width_tol_px=3, dist_kernel=3, width_bin_step=0.2,
    width_peak_prominence=0.5, aspect_ratio_thresh=1.5,
    peak_prominence=10, valley_prominence=5,
    line_color=0,
    detect_L_shapes=True,  # 新增：是否启用L形检测
    L_shape_min_area=200,  # 新增：L形检测的最小面积
    merge_close_rectangles_flag=True,  # 新增：是否合并相近矩形
    merge_distance_factor=2.0):  # 新增：合并距离因子（乘以平均线段宽度）
    
    orig_rgb = cv2.imread(image_path)
    if orig_rgb is None:
        raise ValueError(f"图片读取失败，请检查路径：{image_path}")
    orig_gray = cv2.cvtColor(orig_rgb, cv2.COLOR_BGR2GRAY)
    orig_blur = cv2.GaussianBlur(orig_gray, (3, 3), 0.5)
    h, w = orig_gray.shape
    print("="*150)
    print("🚀 PCB高精度处理流程：直方图分析→芯片处理→元件检测（网格扫描+背景色移除/线段变黑）→黑色处理→引脚颜色转换→修复")
    print("="*150)
    
    # 1. 直方图分析
    print("\n📊 【步骤1：直方图分析】- 确定背景色+黑白阈值")
    hist = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
    hist_ravel = hist.ravel()
    peaks, _ = scipy_find_peaks(hist_ravel, prominence=peak_prominence)
    bg_gray = np.argmax(hist_ravel) if len(peaks)==0 else peaks[np.argmax(hist_ravel[peaks])]
    hist_inv = np.max(hist_ravel) - hist_ravel
    valleys, _ = scipy_find_peaks(hist_inv, prominence=valley_prominence)
    valleys = sorted(valleys)
    if bg_gray > 240:
        black_thresh = max([v for v in valleys if v < bg_gray], default=bg_gray - 10)
        white_thresh = min([v for v in valleys if v > bg_gray], default=min(bg_gray + 2, 255))
    else:
        black_thresh = max([v for v in valleys if v < bg_gray], default=0)
        white_thresh = min([v for v in valleys if v > bg_gray], default=255)
    if white_thresh - black_thresh < 5:
        white_thresh = min(black_thresh + 8, 255)
        black_thresh = max(white_thresh - 8, 0)
    print(f"✅ 直方图结果：背景灰度={bg_gray} | 黑色阈值=<{black_thresh} | 白色阈值=>{white_thresh}")
    
    # 2. 高精度移除芯片主体，保留引脚
    print("\n⚪ 【步骤2：高精度引脚识别+芯片主体移除】")
    # 修复：接收第4个返回值（内部名为chip_area_mask，语义上是芯片主体掩码）
    cluster_removed_gray, cluster_removed_rgb, white_seg_mask, chip_body_mask, pin_mask, stats = accurate_cluster_remove_chip_body_keep_pins(
        orig_gray=orig_blur, orig_rgb=orig_rgb, bg_gray=bg_gray, white_thresh=white_thresh,
        angle_tol=angle_tol, min_area=min_area, dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min_samples,
        pin_angle_tol=pin_angle_tol, pin_convexity_thresh=pin_convexity_thresh, pin_chip_neighbor_thresh=pin_chip_neighbor_thresh,
        chip_body_min_area=chip_body_min_area, chip_aspect_thresh=chip_aspect_thresh, chip_compact_thresh=chip_compact_thresh
    )
    
    mark_vis_rgb = cluster_removed_rgb.copy()
    chip_contours, _ = cv2.findContours(chip_body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mark_vis_rgb, chip_contours, -1, (0,0,255), 2)
    pin_contours, _ = cv2.findContours(pin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mark_vis_rgb, pin_contours, -1, (0,255,0), 3)
    print(f"✅ 芯片主体移除完成：共移除{stats.get('chip_area_pixels', 0)}像素，保留引脚{stats.get('pin_num', 0)}个")
    
    # 3. 【关键集成】使用网格扫描法检测线宽并处理白色区域
    print("\n🔴 【步骤3：元件检测与处理（网格扫描+背景色移除/线段变黑）】")
    component_result = detect_components_from_image(
        img_rgb=cluster_removed_rgb,
        bg_gray=bg_gray,  # 复用芯片处理阶段的背景色
        white_threshold=white_thresh,
        black_threshold=black_thresh,  # 新增：传入黑色阈值用于检测黑色线段
        min_area_ratio=min_area_ratio,
        num_grid_samples=num_grid_samples,
        max_width_ratio=0.1,
        line_color=line_color  # 线段变为此颜色（黑色）
    )
    component_mask = component_result['component_mask']
    line_segment_mask = component_result['line_segment_mask']
    avg_dominant_width = component_result['avg_dominant_width']
    
    # 应用处理结果
    cluster_removed_gray = component_result['result_img_gray']
    cluster_removed_rgb = component_result['result_img_rgb']
    print(f"✅ 白色区域处理完成：")
    print(f"   → 元件移除：{component_result['component_pixels']}像素 → 背景色({bg_gray})")
    print(f"   → 线段保留：{component_result['line_segment_pixels']}像素 → 黑色({line_color})")
    print(f"   → 新增功能：矩形框两端已自动延长连接到黑色线段")
    
    # 4. 初步提取黑色线段（合并原有黑色+新变黑的线段）
    print("\n⚫ 【步骤4：初步提取黑色线段】")
    _, mask_black_orig = cv2.threshold(cluster_removed_gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    mask_black_orig = cv2.morphologyEx(mask_black_orig, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, black_kernel))
    mask_black_orig = cv2.morphologyEx(mask_black_orig, cv2.MORPH_CLOSE, np.ones((1,1), np.uint8), iterations=1)
    
    # 5. 引脚颜色转换
    print("\n🔄 【步骤5：引脚颜色转换】")
    pins_converted_gray, pins_converted_rgb = convert_pins_to_line_color(
        cluster_removed_gray, cluster_removed_rgb, pin_mask, line_color
    )
    print(f"✅ 引脚颜色转换完成：{np.sum(pin_mask > 0)}个引脚像素已转为黑色")
    
    # 6. 黑色线段处理
    print("\n⚫ 【步骤6：黑色线段处理】")
    mask_black_norm, contours_black, corr_cnt = normalize_black_segments(mask_black_orig, angle_tol, black_min_area)
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
    
    # 7. 线段修复
    print("\n🔧 【步骤7：线段修复】")
    # 注意：使用原始白色掩码（含引脚）进行修复
    mask_black_repaired = repair_segments_from_white(
        mask_black_unified, white_seg_mask, contours_black, target_width,
        width_tol_pct, width_tol_px, dist_kernel, aspect_ratio_thresh
    )
    repair_pix = np.sum(mask_black_repaired == 255) - np.sum(mask_black_unified == 255)
    print(f"✅ 线段修复完成：新增修复像素{repair_pix}个")
    
    # 8. 生成最终图像
    print("\n🎨 【步骤8：生成最终图像】")
    final_gray = pins_converted_gray.copy()
    final_rgb = pins_converted_rgb.copy()
    final_gray[mask_black_repaired == 255] = line_color
    if len(final_rgb.shape) == 3:
        for c in range(3):
            final_rgb[:, :, c][mask_black_repaired == 255] = line_color
    else:
        final_rgb[mask_black_repaired == 255] = line_color
    
    # 9. 【新增】对黑色线段进行多边形拟合和矩形标注
    print("\n📐 【步骤9：黑色线段处理（L型多边形拟合 + 矩形标注）】")
    final_with_annotations, shape_info, mask_black_updated = draw_rectangles_and_polygons_on_black_segments(
        final_rgb.copy(), mask_black_repaired, 
        min_area=black_min_area,
        thickness=2,
        detect_L_shapes=detect_L_shapes,
        L_shape_min_area=L_shape_min_area,
        merge_close_rects=merge_close_rectangles_flag,
        avg_line_width=avg_dominant_width,
        line_color=line_color
    )
    
    # 更新最终图像，使用更新后的掩码
    mask_black_repaired = mask_black_updated
    final_gray[mask_black_repaired == 255] = line_color
    if len(final_rgb.shape) == 3:
        for c in range(3):
            final_rgb[:, :, c][mask_black_repaired == 255] = line_color
    else:
        final_rgb[mask_black_repaired == 255] = line_color
    
    # 10. 最终统计
    print(f"\n📊 【最终高精度统计结果】")
    print(f"   → 背景色: {bg_gray}")
    print(f"   → 芯片主体：移除{stats.get('chip_area_pixels', 0)}像素")
    print(f"   → 芯片引脚：保留{stats.get('pin_num', 0)}个")
    print(f"   → 其他元件：移除{component_result['component_pixels']}像素（设为背景色）")
    print(f"   → 线段保留：{component_result['line_segment_pixels']}像素（变为黑色）")
    print(f"   → 线宽检测：网格扫描{num_grid_samples}×{num_grid_samples} | 样本数{len(component_result['line_widths'])} | 主导线宽{component_result['avg_dominant_width']:.2f}px")
    print(f"   → 黑色线段：归一化修正{corr_cnt}个 | 长段{long_cnt} | 短段{short_cnt}")
    print(f"   → 线段修复：修复{repair_pix}像素")
    
    # 统计形状信息 - 修复：使用get方法避免KeyError
    polygon_count = sum(1 for s in shape_info if s.get('is_polygon', False))
    rectangle_count = sum(1 for s in shape_info if not s.get('is_polygon', True))
    merged_count = sum(1 for s in shape_info if s.get('is_merged', False))
    
    print(f"   → 形状标注：共{len(shape_info)}个形状（{polygon_count}个多边形 + {rectangle_count}个矩形）")
    print(f"   → L形检测：{'启用' if detect_L_shapes else '禁用'} | 最小检测面积：{L_shape_min_area}像素")
    print(f"   → L形处理：{polygon_count}个L形状已进行多边形拟合并填充黑色")
    print(f"   → 矩形合并：{'启用' if merge_close_rectangles_flag else '禁用'} | 合并距离：{merge_distance_factor}×线宽")
    
    if merged_count > 0:
        print(f"   → 合并情况：{merged_count}个矩形是通过合并相近矩形得到的")
    
    print("="*150)
    
    # 可视化：12个子图（2行6列）
    plt.figure(figsize=(48, 24), dpi=120)
    
    # ========== 第一行：代码1原有6个图 ==========
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
    
    # ========== 第二行：代码2的6个分析图（含网格扫描+背景色移除+线段变黑） ==========
    plt.subplot(2, 6, 7)
    plt.imshow(cv2.cvtColor(cluster_removed_rgb, cv2.COLOR_BGR2RGB))
    plt.title('7. Component Detection Input', fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.subplot(2, 6, 8)
    hist_ravel = cv2.calcHist([orig_gray], [0], None, [256], [0, 256]).ravel()
    hist_smooth = gaussian_filter1d(hist_ravel, sigma=2)
    plt.plot(hist_ravel, color='black', linewidth=1.5, label='Original histogram')
    plt.plot(hist_smooth, color='red', linewidth=1.5, alpha=0.7, label='Smoothed histogram')
    plt.axvline(x=bg_gray, color='red', linestyle='--', linewidth=2,
                label=f'Background: {bg_gray}')
    plt.axvline(x=white_thresh, color='blue', linestyle='--', linewidth=2,
                label=f'White threshold: {white_thresh}')
    plt.axvline(x=black_thresh, color='green', linestyle='--', linewidth=2,
                label=f'Black threshold: {black_thresh}')
    plt.xlabel('Pixel Intensity', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('8. Histogram with Valley Detection', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.xlim([0, 255])
    
    plt.subplot(2, 6, 9)
    plt.imshow(component_result['white_mask'], cmap='gray')
    white_pix = np.sum(component_result['white_mask'] > 0)
    white_pct = white_pix / (h*w) * 100
    plt.title(f'9. White Regions Mask\n({white_pix:,} pixels, {white_pct:.1f}%)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.subplot(2, 6, 10)
    plt.imshow(cv2.cvtColor(cluster_removed_rgb, cv2.COLOR_BGR2RGB))
    colors = plt.cm.get_cmap('tab20', len(component_result['clustered_regions']))
    for i, region in enumerate(component_result['clustered_regions']):
        x, y, width, height = region['bbox']
        color = colors(i)[:3]
        rect = plt.Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x, y-5, f'{i+1}', fontsize=8, bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    plt.title(f'10. Clustering Result ({len(component_result["clustered_regions"])} clusters)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    # ========== 关键修改：第11图换成带有彩色标注的最终处理结果 ==========
    plt.subplot(2, 6, 11)
    # 使用带有彩色标注的图像
    plt.imshow(cv2.cvtColor(final_with_annotations, cv2.COLOR_BGR2RGB))
    plt.title(f'11. Final Result with Annotations ({len(shape_info)} shapes)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.subplot(2, 6, 12)
    if len(component_result['line_widths']) > 0:
        n, bins, patches = plt.hist(component_result['line_widths'], 
                                   bins=min(30, len(set(component_result['line_widths']))),
                                   color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(component_result['avg_dominant_width'], color='red', linestyle='--', linewidth=2,
                   label=f'Dominant: {component_result["avg_dominant_width"]:.2f}')
        plt.xlabel('Line Width (pixels)', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.title(f'12. Grid Scan Line Width Distribution\n({num_grid_samples}×{num_grid_samples} samples, {len(component_result["line_widths"])} valid)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No valid line widths detected', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=14)
        plt.title('12. Grid Scan Line Width Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
    
    plt.tight_layout()
    save_path = "pcb_complete_processing_grid_scan_bg_removal_line_black.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 高精度结果图保存至：{save_path}")
    
    # 单独保存最终结果图（无标注和有标注两个版本）
    final_save_path = "pcb_final_result.png"
    cv2.imwrite(final_save_path, cv2.cvtColor(final_rgb, cv2.COLOR_BGR2RGB))
    print(f"✅ 最终处理结果（无标注）单独保存至：{final_save_path}")
    
    # 保存带有彩色标注的版本
    final_with_annotations_save_path = "pcb_final_result_with_annotations.png"
    cv2.imwrite(final_with_annotations_save_path, final_with_annotations)
    print(f"✅ 最终处理结果（带彩色标注）单独保存至：{final_with_annotations_save_path}")
    
    return {
        "original_img": {"rgb": orig_rgb, "gray": orig_gray},
        "processed_img": {"rgb": final_rgb, "gray": final_gray},
        "processed_img_with_annotations": final_with_annotations,  # 新增：带彩色标注的图像
        "mask": {
            "chip_body": chip_body_mask,
            "pin": pin_mask,
            "white_seg": white_seg_mask,
            "component": component_mask,
            "line_segment": line_segment_mask,
            "black_orig": mask_black_orig,
            "black_norm": mask_black_norm,
            "black_unified": mask_black_unified,
            "black_repaired": mask_black_repaired
        },
        "shape_info": shape_info,  # 新增：形状信息（包括多边形和矩形）
        "component_detection": component_result,
        "vis_img": {
            "mark_rgb": mark_vis_rgb,
            "validated_rgb": component_result['validated_img']
        },
        "stats": {
            **stats,
            "bg_gray": bg_gray,
            "component_removed_pixels": component_result['component_pixels'],
            "line_segment_pixels": component_result['line_segment_pixels'],
            "component_regions": len(component_result['clustered_regions']),
            "avg_line_width": component_result['avg_dominant_width'],
            "line_width_samples": len(component_result['line_widths']),
            "grid_samples": num_grid_samples,
            "black_long_num": long_cnt,
            "black_short_num": short_cnt,
            "black_corrected_num": corr_cnt,
            "black_target_width": target_width,
            "repair_pixel_num": repair_pix,
            "shape_num": len(shape_info),  # 新增：形状数量
            "polygon_num": sum(1 for s in shape_info if s.get('is_polygon', False)),  # 新增：多边形数量
            "rectangle_num": sum(1 for s in shape_info if not s.get('is_polygon', True)),  # 新增：矩形数量
            "merged_shape_num": sum(1 for s in shape_info if s.get('is_merged', False)),  # 新增：合并形状数量
        },
        "saved_path": save_path,
        "final_result_path": final_save_path,
        "final_with_annotations_path": final_with_annotations_save_path  # 新增：带彩色标注的保存路径
    }

if __name__ == "__main__":
    IMAGE_PATH = r"runs/run18_/segmented_out/middle2__without_segments.jpg"
    try:
        result = process_pcb_segment_accurate_pin_with_components(
            image_path=IMAGE_PATH,
            # 引脚参数：适配低对比度PCB
            pin_angle_tol=12,
            pin_convexity_thresh=0.65,
            pin_chip_neighbor_thresh=25,
            # 芯片参数
            chip_body_min_area=50,
            # 元件检测参数（网格扫描核心）
            min_area_ratio=0.0005,
            num_grid_samples=20,  # 20×20网格扫描
            # 直方图参数
            peak_prominence=5,
            valley_prominence=2,
            # L形检测参数
            detect_L_shapes=True,  # 启用L形检测
            L_shape_min_area=200,  # L形检测的最小面积
            # 矩形合并参数
            merge_close_rectangles_flag=True,  # 启用矩形合并
            merge_distance_factor=2.0,  # 合并距离为2倍线宽
            # 引脚颜色
            line_color=0
        )
        print("\n✅ 处理完成！所有结果已保存。")
        print(f"   → 完整可视化: {result['saved_path']}")
        print(f"   → 最终结果图（无标注）: {result['final_result_path']}")
        print(f"   → 最终结果图（带彩色标注）: {result['final_with_annotations_path']}")
        print(f"   → 保留引脚数: {result['stats']['pin_num']}")
        print(f"   → 移除元件像素: {result['stats']['component_removed_pixels']} (背景色={result['stats']['bg_gray']})")
        print(f"   → 保留线段像素: {result['stats']['line_segment_pixels']} (变为黑色)")
        print(f"   → 网格扫描线宽: {result['stats']['avg_line_width']:.2f}px (基于{result['stats']['line_width_samples']}个样本)")
        print(f"   → 形状标注: {result['stats']['shape_num']}个形状（{result['stats']['polygon_num']}个多边形 + {result['stats']['rectangle_num']}个矩形）")
        print(f"   → L形处理: {result['stats']['polygon_num']}个L形状已进行多边形拟合并填充黑色")
        if result['stats']['merged_shape_num'] > 0:
            print(f"   → 矩形合并: {result['stats']['merged_shape_num']}个矩形是通过合并相近矩形得到的")
    except Exception as e:
        print(f"\n❌ 执行失败：{str(e)}")
        import traceback
        traceback.print_exc()