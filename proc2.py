import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math

def find_width_peak(widths, bin_step=0.2, peak_prominence=0.5):
    """找最右峰值+右侧相邻峰谷（原有逻辑不变）"""
    if len(widths) < 3:
        return np.mean(widths) if widths else 0.0, None, None, None
    
    min_w = max(0, np.min(widths) - 0.5)
    max_w = np.max(widths) + 0.5
    bins = np.arange(min_w, max_w + bin_step, bin_step)
    width_hist, _ = np.histogram(widths, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    peaks, peak_props = find_peaks(width_hist, prominence=peak_prominence)
    if len(peaks) == 0:
        return np.mean(widths), width_hist, bin_centers, None
    
    peak_widths = bin_centers[peaks]
    max_peak_idx = np.argmax(peak_widths)
    max_peak_width = peak_widths[max_peak_idx]
    max_peak_bin_idx = peaks[max_peak_idx]
    
    # 找最右峰值右侧第一个谷值
    right_valley_width = None
    for i in range(max_peak_bin_idx + 1, len(width_hist)):
        left_val = width_hist[i-1]
        current_val = width_hist[i]
        if current_val < left_val:
            if i == len(width_hist) - 1 or current_val < width_hist[i+1]:
                right_valley_width = bin_centers[i]
                break
    if right_valley_width is None:
        right_valley_width = max_w
    
    return max_peak_width, width_hist, bin_centers, right_valley_width

def filter_widths_by_peak(peak_width, right_valley_width, widths, tol_pct=0.2, tol_px=2):
    """仅计算最右峰值-右侧峰谷区间的平均值（原有逻辑不变）"""
    if peak_width == 0 or len(widths) == 0 or right_valley_width is None:
        return widths, np.mean(widths) if widths else 0.0
    
    filtered_widths = []
    for w in widths:
        if peak_width <= w <= right_valley_width:
            pct_error = abs(w - peak_width) / peak_width
            px_error = abs(w - peak_width)
            if pct_error <= tol_pct or px_error <= tol_px:
                filtered_widths.append(w)
    
    if len(filtered_widths) == 0:
        filtered_widths = [w for w in widths if peak_width <= w <= right_valley_width]
        if len(filtered_widths) == 0:
            filtered_widths = widths
    
    filtered_mean = np.mean(filtered_widths)
    return filtered_widths, filtered_mean

def unify_black_width(mask_black, target_width, dist_kernel=3):
    """宽度统一（原有逻辑不变）"""
    dist_transform = cv2.distanceTransform(mask_black, cv2.DIST_L2, dist_kernel)
    max_dist = np.max(dist_transform) if np.max(dist_transform) > 0 else 1
    skeleton = np.zeros_like(mask_black, dtype=np.uint8)
    skeleton[dist_transform >= max_dist * 0.4] = 255
    expand_radius = max(1, int(round(target_width / 2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_radius*2, expand_radius*2))
    mask_black_unified = cv2.dilate(skeleton, kernel, iterations=1)
    mask_black_unified = cv2.bitwise_and(mask_black_unified, mask_black)
    mask_black_unified = cv2.morphologyEx(mask_black_unified, cv2.MORPH_CLOSE, kernel=np.ones((2,2), np.uint8))
    return mask_black_unified

def fit_line_and_get_ends(cnt):
    """拟合线段+提取端点（原有逻辑不变）"""
    [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    cnt_pts = cnt.reshape(-1, 2)
    proj = (cnt_pts[:, 0] - x0) * vx + (cnt_pts[:, 1] - y0) * vy
    p1_idx = np.argmin(proj)
    p2_idx = np.argmax(proj)
    p1 = (int(cnt_pts[p1_idx, 0]), int(cnt_pts[p1_idx, 1]))
    p2 = (int(cnt_pts[p2_idx, 0]), int(cnt_pts[p2_idx, 1]))
    dir_vec = np.array([vx[0], vy[0]])
    dir1 = dir_vec
    dir2 = -dir_vec
    return p1, p2, dir1, dir2

def is_short_segment(cnt, aspect_ratio_thresh=1.5):
    """
    新增：判断是否为短线段（长宽比接近正方形）
    cnt: 线段轮廓
    aspect_ratio_thresh: 长宽比阈值，<1.5判定为短线段（可调整）
    返回：True=短线段，False=长线段
    """
    # 计算包围盒
    x, y, w, h = cv2.boundingRect(cnt)
    # 计算长宽比（取大/小，避免宽>长的情况）
    if min(w, h) == 0:
        return False
    aspect_ratio = max(w, h) / min(w, h)
    # 长宽比<阈值 → 短线段（近乎正方形）
    return aspect_ratio < aspect_ratio_thresh

def explore_four_directions(current_x, current_y, h, w):
    """
    新增：四方向探索（上下左右）
    返回：四个方向的坐标列表
    """
    directions = [
        (current_x + 1, current_y),  # 右
        (current_x - 1, current_y),  # 左
        (current_x, current_y + 1),  # 下
        (current_x, current_y - 1)   # 上
    ]
    # 过滤超出边界的点
    valid_dirs = []
    for (nx, ny) in directions:
        if 0 <= nx < w and 0 <= ny < h:
            valid_dirs.append((nx, ny))
    return valid_dirs

def repair_segments_from_white(mask_black, mask_white, contours_black, target_width, 
                              width_tol_pct=0.2, width_tol_px=2, dist_kernel=3,
                              aspect_ratio_thresh=1.5):
    """
    核心改进：分线段类型探索
    - 长线段：仅沿延长线探索
    - 短线段：上下左右四方向探索
    """
    h, w = mask_black.shape
    mask_repaired = mask_black.copy()
    mask_white_only = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_black))
    visited = np.zeros_like(mask_black, dtype=np.bool_)
    max_extend_step = max(h, w) // 2
    window_size = int(round(target_width * 2))
    half_win = window_size // 2

    for cnt in contours_black:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < 10:
            continue
        
        # 第一步：判断线段类型（长/短）
        is_short = is_short_segment(cnt, aspect_ratio_thresh)
        
        if not is_short:
            # 分支1：长线段 → 沿延长线探索（原有逻辑）
            p1, p2, dir1, dir2 = fit_line_and_get_ends(cnt)
            endpoints = [p1, p2]
            dirs = [dir2, dir1]

            for (ep_x, ep_y), dir_vec in zip(endpoints, dirs):
                if ep_x < 0 or ep_x >= w or ep_y < 0 or ep_y >= h:
                    continue
                current_x, current_y = ep_x, ep_y
                visited[current_y, current_x] = True

                for step in range(max_extend_step):
                    next_x = int(round(current_x + dir_vec[0] * 1))
                    next_y = int(round(current_y + dir_vec[1] * 1))
                    if next_x < 0 or next_x >= w or next_y < 0 or next_y >= h:
                        break
                    if visited[next_y, next_x] or mask_white_only[next_y, next_x] == 0:
                        break
                    visited[next_y, next_x] = True

                    y1 = max(0, next_y - half_win)
                    y2 = min(h, next_y + half_win + 1)
                    x1 = max(0, next_x - half_win)
                    x2 = min(w, next_x + half_win + 1)
                    local_white = mask_white_only[y1:y2, x1:x2]
                    if np.sum(local_white) < 3:
                        break

                    dist_local = cv2.distanceTransform(local_white, cv2.DIST_L2, dist_kernel)
                    foreground_local = dist_local[local_white == 255]
                    local_width = 2 * np.mean(foreground_local) if len(foreground_local) > 0 else 0.0
                    pct_error = abs(local_width - target_width) / target_width if target_width > 0 else 1
                    px_error = abs(local_width - target_width)
                    is_match = (pct_error <= width_tol_pct) or (px_error <= width_tol_px)

                    if not is_match:
                        break

                    mask_repaired[y1:y2, x1:x2] = cv2.bitwise_or(mask_repaired[y1:y2, x1:x2], local_white)
                    current_x, current_y = next_x, next_y
        else:
            # 分支2：短线段 → 上下左右四方向探索
            # 提取短线段的所有边缘点作为起点（避免仅端点探索的局限）
            cnt_mask = np.zeros_like(mask_black)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, 1)  # 仅画轮廓（边缘点）
            edge_pts = np.argwhere(cnt_mask == 255)
            # 遍历所有边缘点
            for (y, x) in edge_pts:
                if visited[y, x]:
                    continue
                # 初始化队列（BFS四方向探索）
                queue = [(x, y)]
                visited[y, x] = True
                # BFS探索
                while queue:
                    cx, cy = queue.pop(0)
                    # 获取四方向有效点
                    valid_dirs = explore_four_directions(cx, cy, h, w)
                    for (nx, ny) in valid_dirs:
                        if visited[ny, nx] or mask_white_only[ny, nx] == 0:
                            continue
                        visited[ny, nx] = True

                        # 局部宽度检测
                        y1 = max(0, ny - half_win)
                        y2 = min(h, ny + half_win + 1)
                        x1 = max(0, nx - half_win)
                        x2 = min(w, nx + half_win + 1)
                        local_white = mask_white_only[y1:y2, x1:x2]
                        if np.sum(local_white) < 3:
                            continue

                        # 宽度匹配
                        dist_local = cv2.distanceTransform(local_white, cv2.DIST_L2, dist_kernel)
                        foreground_local = dist_local[local_white == 255]
                        local_width = 2 * np.mean(foreground_local) if len(foreground_local) > 0 else 0.0
                        pct_error = abs(local_width - target_width) / target_width if target_width > 0 else 1
                        px_error = abs(local_width - target_width)
                        is_match = (pct_error <= width_tol_pct) or (px_error <= width_tol_px)

                        if is_match:
                            # 实心填充
                            mask_repaired[y1:y2, x1:x2] = cv2.bitwise_or(mask_repaired[y1:y2, x1:x2], local_white)
                            # 加入队列继续探索
                            queue.append((nx, ny))

    # 形态学强化（原有逻辑不变）
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask_repaired = cv2.morphologyEx(mask_repaired, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(round(target_width/2)), int(round(target_width/2))))
    mask_repaired = cv2.dilate(mask_repaired, dilate_kernel, iterations=1)
    mask_repaired = cv2.bitwise_and(mask_repaired, cv2.bitwise_or(mask_black, mask_white_only))
    
    return mask_repaired

def process_black_white_segments(image_path,
                                 peak_prominence=15,
                                 valley_prominence=8,
                                 black_min_area=30,
                                 black_kernel=(2,2),
                                 white_min_area=15,
                                 white_open_kernel=(1,1),
                                 white_close_kernel=(2,2),
                                 width_tol_pct=0.2,
                                 width_tol_px=2,
                                 dist_kernel=3,
                                 width_bin_step=0.2,
                                 width_peak_prominence=0.5,
                                 aspect_ratio_thresh=1.5):  # 新增短线段阈值参数
    """主处理函数（新增短线段阈值参数传递）"""
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        raise ValueError(f"图片读取失败，请检查路径：{image_path}")
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0.5)
    h, w = img_gray.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    # 灰度直方图分析（原有逻辑不变）
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_ravel = hist.ravel()
    peaks, _ = find_peaks(hist_ravel, prominence=peak_prominence)
    main_peak = np.argmax(hist_ravel) if len(peaks) == 0 else peaks[np.argmax(hist_ravel[peaks])]
    hist_inv = np.max(hist_ravel) - hist_ravel
    valleys, _ = find_peaks(hist_inv, prominence=valley_prominence)
    valleys = sorted(valleys)
    left_valley = max([v for v in valleys if v < main_peak], default=0)
    right_valley = min([v for v in valleys if v > main_peak], default=255)
    bg_gray_range = [left_valley, right_valley]
    print("="*70)
    print(f"📌 背景主峰值：{main_peak} | 背景区间：[{left_valley}, {right_valley}]")

    # 黑色线段提取（原有逻辑不变）
    _, mask_black = cv2.threshold(img_blur, left_valley, 255, cv2.THRESH_BINARY_INV)
    kernel_black = cv2.getStructuringElement(cv2.MORPH_RECT, black_kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, black_kernel)
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_black_final = np.zeros_like(img_gray)
    black_widths = []
    # 新增：统计长/短线段数量
    long_segment_count = 0
    short_segment_count = 0
    for cnt in contours_black:
        if cv2.contourArea(cnt) >= black_min_area:
            cnt_mask = np.zeros_like(img_gray)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            dist_cnt = cv2.distanceTransform(cnt_mask, cv2.DIST_L2, dist_kernel)
            foreground_cnt = dist_cnt[cnt_mask == 255]
            cnt_width = 2 * np.mean(foreground_cnt) if len(foreground_cnt) > 0 else 0.0
            if cnt_width > 0:
                black_widths.append(cnt_width)
            cv2.drawContours(mask_black_final, [cnt], -1, 255, -1)
            # 统计长/短线段
            if is_short_segment(cnt, aspect_ratio_thresh):
                short_segment_count += 1
            else:
                long_segment_count += 1
    mask_black_final = cv2.morphologyEx(mask_black_final, cv2.MORPH_CLOSE, black_kernel)
    # 打印长/短线段统计
    print(f"📏 黑色线段统计：总长段{long_segment_count}根 | 总短段{short_segment_count}根")
    
    # 宽度峰值+区间平均（原有逻辑不变）
    max_peak_width, width_hist, bin_centers, right_valley_width = find_width_peak(
        black_widths, bin_step=width_bin_step, peak_prominence=width_peak_prominence
    )
    filtered_widths, target_unify_width = filter_widths_by_peak(
        max_peak_width, right_valley_width, black_widths, tol_pct=width_tol_pct, tol_px=width_tol_px
    )
    print(f"📏 宽度统计（仅最右峰值-右侧峰谷区间）：")
    print(f"   最右峰值宽度：{max_peak_width:.2f}px | 右侧相邻峰谷宽度：{right_valley_width:.2f}px")
    print(f"   区间内宽度数量：{len(filtered_widths)} | 区间内平均值：{target_unify_width:.2f}px")
    print(f"   原始所有宽度范围：{np.min(black_widths):.2f}px ~ {np.max(black_widths):.2f}px（仅参考）")
    mask_black_unified = unify_black_width(mask_black_final, target_unify_width, dist_kernel)
    print(f"✅ 黑色线段宽度统一完成 | 统一为区间平均值：{target_unify_width:.2f}px")
    print(f"🎯 宽度匹配阈值：±{width_tol_pct*100}% 或 ±{width_tol_px}px")
    print("="*70)

    # 白色线段提取（原有逻辑不变）
    _, mask_white = cv2.threshold(img_blur, right_valley, 255, cv2.THRESH_BINARY)
    kernel_white_close = cv2.getStructuringElement(cv2.MORPH_RECT, white_close_kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, white_close_kernel)
    kernel_white_open = cv2.getStructuringElement(cv2.MORPH_RECT, white_open_kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, white_open_kernel)
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_white_final = np.zeros_like(img_gray)
    for cnt in contours_white:
        if cv2.contourArea(cnt) >= white_min_area:
            cv2.drawContours(mask_white_final, [cnt], -1, 255, -1)
    mask_white_final = cv2.morphologyEx(mask_white_final, cv2.MORPH_CLOSE, (2,2))

    # 线段修复：传递短线段阈值参数
    mask_repaired = repair_segments_from_white(
        mask_black_unified, mask_white_final, contours_black, target_unify_width,
        width_tol_pct, width_tol_px, dist_kernel, aspect_ratio_thresh
    )
    print(f"✅ 线段修复完成（长段沿延长线/短段四方向） | 修复后黑色像素数：{np.sum(mask_repaired == 255)}")

    # 白色线段标红（原有逻辑不变）
    white_segment_info = []
    mask_white_debug = mask_white_final.copy()
    img_white_match_red = img_rgb.copy()
    mask_white_match = np.zeros_like(img_gray)
    if np.sum(mask_white_final) > 0 and len(contours_white) > 0:
        dist_white = cv2.distanceTransform(mask_white_final, cv2.DIST_L2, dist_kernel)
        for idx, cnt in enumerate(contours_white):
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < white_min_area:
                continue
            cnt_mask = np.zeros_like(img_gray)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            cnt_dist = dist_white[cnt_mask == 255]
            cnt_avg_width = 2 * np.mean(cnt_dist) if len(cnt_dist) > 0 else 0.0
            pct_error = abs(cnt_avg_width - target_unify_width) / target_unify_width if target_unify_width > 0 else 1
            px_error = abs(cnt_avg_width - target_unify_width)
            is_match = (pct_error <= width_tol_pct) or (px_error <= width_tol_px)
            x, y, cw, ch = cv2.boundingRect(cnt)
            white_segment_info.append({
                "idx": idx+1, "area": cnt_area, "avg_width": cnt_avg_width,
                "pct_error": pct_error, "px_error": px_error, "is_match": is_match,
                "contour": cnt, "x":x, "y":y
            })
            cv2.putText(mask_white_debug, f"{cnt_avg_width:.1f}px", (x+2, y+12),
                        font, font_scale, 255, font_thickness)
            if is_match:
                cv2.drawContours(mask_white_match, [cnt], -1, 255, -1)
                cv2.drawContours(img_white_match_red, [cnt], -1, (0, 0, 255), -1)

    # 统计打印（原有逻辑不变）
    white_total = len(white_segment_info)
    white_match = sum([1 for seg in white_segment_info if seg["is_match"]])
    print(f"📊 白色线段统计：总{white_total}根 | 宽度匹配{white_match}根（标红）")
    if white_total > 0:
        print("🔍 白色线段明细（宽度/误差/是否匹配）：")
        for seg in white_segment_info:
            print(f"   线段{seg['idx']}：{seg['avg_width']:.2f}px | 误差{seg['pct_error']*100:.1f}%/{seg['px_error']:.1f}px | {seg['is_match']}")
    print("="*70)

    # 可视化（新增长/短线段标注）
    plt.figure(figsize=(36, 12), dpi=120)
    plt.subplot(2, 5, 1)
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.title('1. Original Image', fontsize=12)
    plt.axis('off')
    plt.subplot(2, 5, 2)
    plt.imshow(mask_black_final, cmap='gray')
    plt.title(f'2. Original Black Segments\nLong: {long_segment_count} | Short: {short_segment_count}', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 5, 3)
    if width_hist is not None and bin_centers is not None:
        plt.plot(bin_centers, width_hist, color='black', linewidth=2)
        plt.axvline(max_peak_width, color='red', linestyle='--', linewidth=2, label=f'Right Peak: {max_peak_width:.2f}px')
        plt.axvline(right_valley_width, color='green', linestyle='--', linewidth=2, label=f'Right Valley: {right_valley_width:.2f}px')
        plt.axvspan(max_peak_width, right_valley_width, color='orange', alpha=0.2, label='Avg Calculation Range')
        plt.axvline(target_unify_width, color='blue', linestyle='-', linewidth=2, label=f'Target Avg: {target_unify_width:.2f}px')
        plt.legend(fontsize=8)
    plt.xlabel('Width (px)', fontsize=9)
    plt.ylabel('Count', fontsize=9)
    plt.title('3. Black Width Distribution (Avg Range Marked)', fontsize=10)
    plt.grid(alpha=0.3)
    plt.subplot(2, 5, 4)
    plt.imshow(mask_black_unified, cmap='gray')
    plt.title(f'4. Unified Black Segments\nWidth: {target_unify_width:.2f}px', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 5, 5)
    plt.imshow(mask_repaired, cmap='gray')
    plt.title('5. Repaired Segments (Long:ExtLine | Short:4Dir)', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 5, 6)
    plt.imshow(mask_white_debug, cmap='gray')
    plt.title(f'6. White Segments\nTotal: {white_total}', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 5, 7)
    plt.plot(hist_ravel, color='black', linewidth=1)
    plt.fill_between(range(256), hist_ravel, color='gray', alpha=0.7)
    plt.scatter(main_peak, hist_ravel[main_peak], color='red', s=60, label=f'BG Peak: {main_peak}')
    plt.scatter(left_valley, hist_ravel[left_valley], color='blue', s=60)
    plt.scatter(right_valley, hist_ravel[right_valley], color='blue', s=60)
    plt.axvspan(bg_gray_range[0], bg_gray_range[1], color='green', alpha=0.2)
    plt.axvspan(0, left_valley, color='cyan', alpha=0.2)
    plt.axvspan(right_valley, 255, color='red', alpha=0.2)
    plt.xlabel('Grayscale', fontsize=9)
    plt.ylabel('Count', fontsize=9)
    plt.xlim(0, 255)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)
    plt.title('7. Grayscale Histogram', fontsize=10)
    plt.subplot(2, 5, 8)
    img_all_red = img_rgb.copy()
    img_all_red[mask_black_final == 255] = [0,0,255]
    img_all_red[mask_white_final == 255] = [0,0,255]
    plt.imshow(cv2.cvtColor(img_all_red, cv2.COLOR_BGR2RGB))
    plt.title('8. All Segments (Red)', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 5, 9)
    plt.imshow(cv2.cvtColor(img_white_match_red, cv2.COLOR_BGR2RGB))
    plt.title(f'9. Matched White Seg (Red)\n{white_match}/{white_total}', fontsize=10)
    plt.axis('off')
    plt.subplot(2, 5, 10)
    plt.text(0.5, 0.5, 'PCB Segment Repair\nLong: Extension Line | Short: 4 Directions', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = "pcb_segment_repair_long_short_adaptive.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return {
        "bg_info": {"peak":main_peak, "left_v":left_valley, "right_v":right_valley, "range":bg_gray_range},
        "black": {
            "total_count": len(black_widths),
            "long_count": long_segment_count,
            "short_count": short_segment_count,
            "width_range": (np.min(black_widths), np.max(black_widths)) if black_widths else (0,0),
            "original_avg": np.mean(black_widths) if black_widths else 0.0,
            "peak_width": max_peak_width,
            "right_valley_width": right_valley_width,
            "filtered_count": len(filtered_widths),
            "target_width": target_unify_width,
            "original_mask": mask_black_final,
            "unified_mask": mask_black_unified,
            "repaired_mask": mask_repaired
        },
        "white": {"total":white_total, "match":white_match, "info":white_segment_info, "mask":mask_white_final, "match_mask":mask_white_match},
        "red_img": {"all":img_all_red, "match_white":img_white_match_red},
        "saved_path": save_path
    }

# 主函数调用
if __name__ == "__main__":
    # 替换为你的PCB线段图片路径
    IMAGE_PATH = r"runs/run15/segmented_out/complex_without_segments.jpg"
    try:
        print("🚀 开始PCB线段处理（长段沿延长线/短段四方向）...")
        result = process_black_white_segments(
            image_path=IMAGE_PATH,
            peak_prominence=12,
            valley_prominence=6,
            black_min_area=25,
            white_min_area=10,
            width_tol_pct=0.2,
            width_tol_px=2,
            width_bin_step=0.2,
            width_peak_prominence=0.5,
            aspect_ratio_thresh=1.5  # 短线段阈值：长宽比<1.5判定为短线段
        )
        print(f"\n✅ 处理完成！结果已保存至：{result['saved_path']}")
        print(f"📌 最终目标宽度（区间平均）：{result['black']['target_width']:.2f}px")
        print(f"   长线段：{result['black']['long_count']}根 | 短线段：{result['black']['short_count']}根")
    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")
        import traceback
        traceback.print_exc()