import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def process_black_white_segments(image_path,
                                 # 直方图峰谷检测参数
                                 peak_prominence=15,
                                 valley_prominence=8,
                                 # 黑色线段提取参数
                                 black_min_area=30,
                                 black_kernel=(2,2),
                                 # 白色线段提取参数（单独调，避免过度过滤）
                                 white_min_area=15,  # 降低，保留细线段
                                 white_open_kernel=(1,1),  # 减小，少去噪
                                 white_close_kernel=(2,2), # 小核修复裂痕
                                 # 宽度匹配双阈值（核心：解决漏标）
                                 width_tol_pct=0.2,   # 百分比误差（20%）
                                 width_tol_px=2,      # 绝对像素误差（2px）
                                 # 距离变换核（调小，提升细线段精度）
                                 dist_kernel=3):
    """
    解决白色线段标红漏标问题：精准宽度计算+双阈值匹配+松限制提取
    保留原有4张核心图，新增调试标注+精准标红图，自动保存结果
    """
    # ===================== 1. 图像读取与预处理 =====================
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        raise ValueError(f"图片读取失败，请检查路径：{image_path}")
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # 轻量高斯滤波（σ=0.5，仅去噪不模糊细线段）
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0.5)
    h, w = img_gray.shape
    # 生成画布用于标注文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1

    # ===================== 2. 直方图分析：峰值+左右峰谷界定背景（原有逻辑保留） =====================
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_ravel = hist.ravel()
    # 检测主峰值
    peaks, _ = find_peaks(hist_ravel, prominence=peak_prominence)
    main_peak = np.argmax(hist_ravel) if len(peaks) == 0 else peaks[np.argmax(hist_ravel[peaks])]
    # 检测波谷并找左右相邻
    hist_inv = np.max(hist_ravel) - hist_ravel
    valleys, _ = find_peaks(hist_inv, prominence=valley_prominence)
    valleys = sorted(valleys)
    left_valley = max([v for v in valleys if v < main_peak], default=0)
    right_valley = min([v for v in valleys if v > main_peak], default=255)
    bg_gray_range = [left_valley, right_valley]
    # 终端打印基础信息
    print("="*60)
    print(f"📌 背景主峰值：{main_peak} | 背景区间：[{left_valley}, {right_valley}]")

    # ===================== 3. 黑色线段提取+精准宽度计算（原有逻辑优化，保留） =====================
    _, mask_black = cv2.threshold(img_blur, left_valley, 255, cv2.THRESH_BINARY_INV)
    kernel_black = cv2.getStructuringElement(cv2.MORPH_RECT, black_kernel)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel_black)
    # 找连通域过滤
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_black_final = np.zeros_like(img_gray)
    for cnt in contours_black:
        if cv2.contourArea(cnt) >= black_min_area:
            cv2.drawContours(mask_black_final, [cnt], -1, 255, -1)
    mask_black_final = cv2.morphologyEx(mask_black_final, cv2.MORPH_CLOSE, kernel_black)
    # 精准计算黑色平均宽度（距离变换+前景像素，无骨架依赖）
    black_avg_width = 0.0
    if np.sum(mask_black_final) > 0:
        dist_black = cv2.distanceTransform(mask_black_final, cv2.DIST_L2, dist_kernel)
        foreground_black = dist_black[mask_black_final == 255]
        black_avg_width = 2 * np.mean(foreground_black)
    print(f"✅ 黑色线段提取完成 | 平均宽度：{black_avg_width:.2f} 像素")
    print(f"🎯 白色线段匹配阈值：±{width_tol_pct*100}% 或 ±{width_tol_px} 像素")
    print("="*60)

    # ===================== 4. 白色线段提取：松限制+先修复+保留所有有效线段（核心优化） =====================
    # 步骤4.1：二值化提取白色线段
    _, mask_white = cv2.threshold(img_blur, right_valley, 255, cv2.THRESH_BINARY)
    # 步骤4.2：先闭运算修复微小裂痕（避免完整线段拆分成小连通域）
    kernel_white_close = cv2.getStructuringElement(cv2.MORPH_RECT, white_close_kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel_white_close)
    # 步骤4.3：小核开运算轻微去噪（仅过滤1像素噪点）
    kernel_white_open = cv2.getStructuringElement(cv2.MORPH_RECT, white_open_kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel_white_open)
    # 步骤4.4：找连通域，极低面积阈值保留细线段
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_white_final = np.zeros_like(img_gray)
    # 先绘制所有白色线段掩码（用于后续宽度计算）
    for cnt in contours_white:
        if cv2.contourArea(cnt) >= white_min_area:
            cv2.drawContours(mask_white_final, [cnt], -1, 255, -1)
    # 轻微闭运算修复宽度计算时的边缘噪点
    mask_white_final = cv2.morphologyEx(mask_white_final, cv2.MORPH_CLOSE, (2,2))

    # ===================== 5. 白色线段：逐根精准计算宽度+双阈值匹配（核心优化） =====================
    white_segment_info = []  # 存储每根线段的完整信息
    mask_white_debug = mask_white_final.copy()  # 调试掩码：标注宽度
    img_white_match_red = img_rgb.copy()        # 最终标红图：仅匹配的白色线段
    mask_white_match = np.zeros_like(img_gray)  # 匹配的白色线段掩码
    # 仅当有白色线段时计算宽度
    if np.sum(mask_white_final) > 0 and len(contours_white) > 0:
        # 距离变换（和黑色线段同参数，保证精度一致）
        dist_white = cv2.distanceTransform(mask_white_final, cv2.DIST_L2, dist_kernel)
        # 逐根处理白色线段
        for idx, cnt in enumerate(contours_white):
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < white_min_area:
                continue
            # 生成单根线段的掩码
            cnt_mask = np.zeros_like(img_gray)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            # 精准计算该线段的平均宽度（和黑色线段完全一致的逻辑）
            cnt_dist = dist_white[cnt_mask == 255]
            cnt_avg_width = 2 * np.mean(cnt_dist) if len(cnt_dist) > 0 else 0.0
            # 双阈值判断是否匹配（核心：解决漏标）
            pct_error = abs(cnt_avg_width - black_avg_width) / black_avg_width if black_avg_width > 0 else 1
            px_error = abs(cnt_avg_width - black_avg_width)
            is_match = (pct_error <= width_tol_pct) or (px_error <= width_tol_px)
            # 记录信息
            x, y, cw, ch = cv2.boundingRect(cnt)
            white_segment_info.append({
                "idx": idx+1,
                "area": cnt_area,
                "avg_width": cnt_avg_width,
                "pct_error": pct_error,
                "px_error": px_error,
                "is_match": is_match,
                "contour": cnt,
                "x":x, "y":y
            })
            # 调试标注：在白色线段掩码图上写宽度（左上角）
            cv2.putText(mask_white_debug, f"{cnt_avg_width:.1f}px", (x+2, y+12),
                        font, font_scale, 255, font_thickness)
            # 匹配则标红+绘制掩码
            if is_match:
                cv2.drawContours(mask_white_match, [cnt], -1, 255, -1)
                cv2.drawContours(img_white_match_red, [cnt], -1, (0, 0, 255), -1)

    # ===================== 6. 统计结果并终端打印明细 =====================
    white_total = len(white_segment_info)
    white_match = sum([1 for seg in white_segment_info if seg["is_match"]])
    print(f"📊 白色线段统计：总{white_total}根 | 匹配{white_match}根（标红）")
    if white_total > 0:
        print("🔍 白色线段明细（宽度/误差/是否匹配）：")
        for seg in white_segment_info:
            print(f"   线段{seg['idx']}：{seg['avg_width']:.2f}px | 误差{seg['pct_error']*100:.1f}%/{seg['px_error']:.1f}px | {seg['is_match']}")
    print("="*60)

    # ===================== 7. 可视化：保留原有4图 + 新增2张调试/标红图（共6图） =====================
    plt.figure(figsize=(24, 10), dpi=120)
    # 子图1：原始图片（原有）
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.title('1. Original Image', fontsize=12)
    plt.axis('off')
    # 子图2：黑色线段（原有）
    plt.subplot(2, 3, 2)
    plt.imshow(mask_black_final, cmap='gray')
    plt.title(f'2. Black Segments\nAvg Width: {black_avg_width:.2f}px', fontsize=10)
    plt.axis('off')
    # 子图3：白色线段+宽度标注（调试版，原有基础优化）
    plt.subplot(2, 3, 3)
    plt.imshow(mask_white_debug, cmap='gray')
    plt.title(f'3. White Segments (Width Labeled)\nTotal: {white_total}', fontsize=10)
    plt.axis('off')
    # 子图4：灰度直方图（原有，完整保留标注）
    plt.subplot(2, 3, 4)
    plt.plot(hist_ravel, color='black', linewidth=1)
    plt.fill_between(range(256), hist_ravel, color='gray', alpha=0.7)
    plt.scatter(main_peak, hist_ravel[main_peak], color='red', s=60, label=f'BG Peak: {main_peak}')
    plt.scatter(left_valley, hist_ravel[left_valley], color='blue', s=60, label=f'Left Valley: {left_valley}')
    plt.scatter(right_valley, hist_ravel[right_valley], color='blue', s=60, label=f'Right Valley: {right_valley}')
    plt.axvspan(bg_gray_range[0], bg_gray_range[1], color='green', alpha=0.2, label=f'BG Range: {bg_gray_range}')
    plt.axvspan(0, left_valley, color='cyan', alpha=0.2, label='Black Seg')
    plt.axvspan(right_valley, 255, color='red', alpha=0.2, label='White Seg')
    plt.xlabel('Grayscale Value', fontsize=9)
    plt.ylabel('Pixel Count', fontsize=9)
    plt.xlim(0, 255)
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)
    plt.title('4. Grayscale Histogram (BG: Peak+2Valleys)', fontsize=10)
    # 子图5：所有黑白线段标红（原有）
    plt.subplot(2, 3, 5)
    img_all_red = img_rgb.copy()
    img_all_red[mask_black_final == 255] = [0,0,255]
    img_all_red[mask_white_final == 255] = [0,0,255]
    plt.imshow(cv2.cvtColor(img_all_red, cv2.COLOR_BGR2RGB))
    plt.title('5. All Segments (Red)', fontsize=10)
    plt.axis('off')
    # 子图6：仅宽度匹配的白色线段标红（最终需求图）
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(img_white_match_red, cv2.COLOR_BGR2RGB))
    plt.title(f'6. Matched White Seg (Red)\nMatch: {white_match}/{white_total}', fontsize=10)
    plt.axis('off')
    # 调整布局并保存图片
    plt.tight_layout()
    save_path = "segment_matching_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # ===================== 返回所有结果 =====================
    return {
        "bg_info": {"peak":main_peak, "left_v":left_valley, "right_v":right_valley, "range":bg_gray_range},
        "black": {"avg_width":black_avg_width, "mask":mask_black_final},
        "white": {"total":white_total, "match":white_match, "info":white_segment_info, "mask":mask_white_final, "match_mask":mask_white_match},
        "red_img": {"all":img_all_red, "match_white":img_white_match_red},
        "saved_path": save_path
    }

# ===================== 主函数调用 =====================
if __name__ == "__main__":
    # 替换为你的图片路径
    IMAGE_PATH = r"runs/run15/segmented_out/complex_without_segments.jpg"
    try:
        result = process_black_white_segments(
            image_path=IMAGE_PATH,
            peak_prominence=12,
            valley_prominence=6,
            black_min_area=25,
            white_min_area=10,  # 极低阈值，保留极细线段
            width_tol_pct=0.2,  # 20%百分比误差
            width_tol_px=2      # 2px绝对像素误差
        )
        print(f"\n✅ 处理完成！结果图片已保存至：{result['saved_path']}")
    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")