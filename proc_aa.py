import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_with_expansion_by_line_width(image_path):
    """
    1. 检测封闭区域
    2. 统计线段宽度
    3. 将封闭区域外扩 (线段宽度 + 1) 像素
    4. 外扩区域置为白色
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. 提取黑色线段
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 2. 【统计线段宽度】通过形态学操作估计
    # 腐蚀操作，直到大部分线段消失，记录腐蚀次数
    eroded = black_mask.copy()
    erosion_count = 0
    while np.any(eroded > 0):
        eroded = cv2.erode(eroded, np.ones((3,3), np.uint8), iterations=1)
        erosion_count += 1
        if erosion_count > 20:  # 防止无限循环
            break
    
    # 线段宽度约为腐蚀次数的2倍（因为从两边腐蚀）
    estimated_line_width = max(1, erosion_count * 2 - 2)
    print(f"✓ 估计线段宽度: {estimated_line_width} 像素")
    
    # 3. 【检测封闭区域】使用洪水填充法
    h, w = black_mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 创建反向掩膜（白色区域变黑，黑色区域变白）
    inverted_mask = cv2.bitwise_not(black_mask)
    
    # 从边缘开始洪水填充所有可达的白色区域（背景）
    for seed in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
        x, y = seed
        if inverted_mask[y, x] == 255:  # 白色背景
            temp_mask = inverted_mask.copy()
            cv2.floodFill(temp_mask, flood_mask, seed, 128,
                         loDiff=0, upDiff=0, flags=4 | (255 << 8))
    
    # 提取封闭区域
    reachable_bg = flood_mask[1:-1, 1:-1].copy()
    reachable_bg = (reachable_bg > 0).astype(np.uint8) * 255
    original_white = (black_mask == 0).astype(np.uint8) * 255
    enclosed_areas = cv2.bitwise_and(original_white, cv2.bitwise_not(reachable_bg))
    
    # 4. 【验证封闭区域】检查是否主要由横平竖直线段组成
    contours, _ = cv2.findContours(enclosed_areas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_enclosed_regions = np.zeros_like(enclosed_areas)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # 过滤小噪声
            continue
            
        # 检查轮廓的形状特征
        if len(contour) >= 4:
            # 计算轮廓的方向分布
            horizontal_count = 0
            vertical_count = 0
            total_count = 0
            
            points = contour.reshape(-1, 2)
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])
                
                if dx > dy and dx > 2:  # 水平线段
                    horizontal_count += 1
                elif dy > dx and dy > 2:  # 垂直线段
                    vertical_count += 1
                total_count += 1
            
            # 如果主要是横平竖直线段组成的区域
            hor_ver_ratio = (horizontal_count + vertical_count) / max(total_count, 1)
            if hor_ver_ratio > 0.7:  # 70%以上是横平竖直线段
                cv2.fillPoly(valid_enclosed_regions, [contour], 255)
    
    # 5. 【外扩处理】
    # 计算外扩半径：线段宽度 + 1
    expansion_radius = estimated_line_width + 1
    
    # 创建外扩的核
    kernel_size = 2 * expansion_radius + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # 对有效封闭区域进行膨胀（外扩）
    expanded_mask = cv2.dilate(valid_enclosed_regions, kernel, iterations=1)
    
    # 6. 【应用处理】将外扩区域置为白色
    result[expanded_mask > 0] = [255, 255, 255]
    
    # 7. 统计结果
    final_contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_count = len(final_contours)
    
    # 8. 可视化
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20, 4))
    
    plt.subplot(1, 6, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 6, 2)
    plt.imshow(black_mask, cmap='gray')
    plt.title('Black Lines Mask')
    plt.axis('off')
    
    plt.subplot(1, 6, 3)
    plt.imshow(enclosed_areas, cmap='hot')
    plt.title(f'All Enclosed Areas\n(Total: {len(contours)})')
    plt.axis('off')
    
    plt.subplot(1, 6, 4)
    plt.imshow(valid_enclosed_regions, cmap='hot')
    plt.title(f'Valid Rectangular\nAreas: {len(cv2.findContours(valid_enclosed_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])}')
    plt.axis('off')
    
    plt.subplot(1, 6, 5)
    plt.imshow(expanded_mask, cmap='hot')
    plt.title(f'Expanded Mask\n(Radius: {expansion_radius})')
    plt.axis('off')
    
    plt.subplot(1, 6, 6)
    plt.imshow(result_rgb)
    plt.title('Final Result\n(Expanded areas → White)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ 处理完成：线段宽度估计 {estimated_line_width} 像素")
    print(f"✓ 外扩半径: {expansion_radius} 像素")
    print(f"✓ 处理了 {processed_count} 个封闭区域（包含外扩）")
    print("✓ 外扩区域已置为白色，其他内容保持不变")
    
    return processed_count, result_rgb

# 执行处理
if __name__ == "__main__":
    image_path = "runs/run18_4/segmented_out/simple_without_segments.jpg"
    try:
        count, processed_img = process_with_expansion_by_line_width(image_path)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()