import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from collections import defaultdict

def detect_and_mark_white_regions(image_path, white_threshold=None, min_area_ratio=0.0005):
    """
    检测白色区域并用红色矩形框标记
    
    步骤：
    1. 读取图像并转换为灰度
    2. 直方图分析
    3. 提取白色区域（自动设置阈值为背景色右边的第一个波谷）
    """
    
    print(f"Processing image: {image_path}")
    
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image from path: {image_path}")
    
    # 转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    total_pixels = h * w
    
    print(f"Image size: {w}x{h} ({total_pixels:,} pixels)")
    
    # 2. 直方图统计
    print("Calculating histogram...")
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist_ravel = hist.ravel()
    
    # 找到峰值
    peaks, _ = find_peaks(hist_ravel, prominence=10)
    if len(peaks) > 0:
        bg_gray = peaks[np.argmax(hist_ravel[peaks])]
    else:
        bg_gray = np.argmax(hist_ravel)
    
    print(f"Background peak at gray level: {bg_gray}")
    
    # 3. 自动确定白色阈值（背景色右边的第一个波谷）
    if white_threshold is None:
        print("Automatically determining white threshold...")
        
        # 对直方图进行平滑处理，便于找到波谷
        hist_smooth = gaussian_filter1d(hist_ravel, sigma=2)
        
        # 寻找波谷：局部最小值
        # 我们只关心背景峰右侧的波谷
        search_start = int(bg_gray)
        search_range = hist_smooth[search_start:]
        
        # 计算导数符号变化
        diff = np.diff(search_range)
        sign_change = np.where(np.diff(np.sign(diff)) > 0)[0]
        
        if len(sign_change) > 0:
            # 找到第一个波谷（从背景峰开始）
            first_valley = search_start + sign_change[0]
            white_threshold = int(first_valley)
            
            # 确保阈值在合理范围内
            if white_threshold < bg_gray + 10:
                white_threshold = min(bg_gray + 30, 250)
            elif white_threshold > 250:
                white_threshold = 240
        else:
            # 如果没有找到明显的波谷，使用启发式方法
            # 通常白色阈值比背景峰高30-50
            white_threshold = min(bg_gray + 40, 240)
            
        print(f"Auto-determined white threshold: {white_threshold}")
    else:
        print(f"Using manual white threshold: {white_threshold}")
    
    # 提取白色区域
    print(f"Extracting white regions with threshold: {white_threshold}")
    
    # 二值化提取白色区域
    _, white_mask = cv2.threshold(img_gray, white_threshold, 255, cv2.THRESH_BINARY)
    
    # 形态学操作清理掩码
    kernel_size = max(1, min(h, w) // 200)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # 4. 检测白色区域轮廓
    print("Detecting white regions...")
    
    # 查找轮廓
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions_info = []
    
    # 设置最小面积阈值
    min_area = max(20, int(total_pixels * min_area_ratio))
    print(f"Minimum area for region detection: {min_area} pixels")
    
    # 遍历轮廓并统计
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # 跳过太小的区域
        if area < min_area:
            continue
        
        # 获取边界矩形
        x, y, width, height = cv2.boundingRect(cnt)
        
        # 存储区域信息
        regions_info.append({
            'id': i + 1,
            'area': area,
            'bbox': (x, y, width, height),
            'aspect_ratio': max(width, height) / max(min(width, height), 1),
        })
    
    print(f"Detected {len(regions_info)} white regions")
    
    # 5. 使用聚类方法重新组织白色像素
    print("Applying clustering to white regions...")
    
    # 获取所有白色像素的坐标
    white_coords = np.column_stack(np.where(white_mask > 0))
    
    if len(white_coords) > 0:
        # 使用DBSCAN聚类算法对白色像素进行分组
        clustering = DBSCAN(eps=10, min_samples=5)  # eps控制聚类距离，min_samples控制最小样本数
        cluster_labels = clustering.fit_predict(white_coords)
        
        # 将聚类结果转换回图像形式
        clustered_mask = np.zeros_like(white_mask)
        for idx, coord in enumerate(white_coords):
            row, col = coord
            cluster_id = cluster_labels[idx]
            if cluster_id != -1:  # -1表示噪声点
                clustered_mask[row, col] = cluster_id + 1
        
        # 获取聚类后的新区域信息
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)  # 移除噪声点
        
        clustered_regions_raw = []
        for cluster_id in unique_clusters:
            # 获取当前聚类的所有像素坐标
            cluster_coords = white_coords[cluster_labels == cluster_id]
            
            # 计算边界框
            y_coords = cluster_coords[:, 0]
            x_coords = cluster_coords[:, 1]
            
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # 计算面积
            area = len(cluster_coords)
            
            # 计算宽高比
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            
            clustered_regions_raw.append({
                'id': cluster_id + 1,
                'area': area,
                'bbox': (min_x, min_y, width, height),
                'aspect_ratio': aspect_ratio,
            })
        
        print(f"Initially clustered into {len(clustered_regions_raw)} regions")
        
        # 过滤掉接近整张图片尺寸的矩形框
        filtered_clustered_regions = []
        for region in clustered_regions_raw:
            x, y, width, height = region['bbox']
            
            # 检查矩形是否接近整张图片尺寸（例如，超过原图的80%）
            width_ratio = width / w
            height_ratio = height / h
            
            # 如果矩形尺寸接近整张图片尺寸，则跳过
            if width_ratio > 0.8 or height_ratio > 0.8:
                print(f"Filtering out region {region['id']}: too close to image boundaries "
                      f"(width_ratio: {width_ratio:.2f}, height_ratio: {height_ratio:.2f})")
                continue
            else:
                filtered_clustered_regions.append(region)
        
        clustered_regions = filtered_clustered_regions
        print(f"After filtering: {len(clustered_regions)} regions remaining")
        
    else:
        clustered_regions = []
        clustered_mask = np.zeros_like(white_mask)
    
    # 6. 专门检测水平和垂直黑色线段
    print("Detecting horizontal and vertical black line segments...")
    
    # 创建二值图像，黑色为前景（255），白色为背景（0）
    black_thresh = bg_gray  # 使用背景灰度值作为黑色阈值
    _, binary_inv = cv2.threshold(img_gray, black_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # 找到黑色像素的坐标
    black_coords = np.column_stack(np.where(binary_inv > 0))
    
    line_widths = []
    
    if len(black_coords) > 0:
        # 使用形态学操作分离水平和垂直线段
        # 先对图像进行一些清理
        cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        
        # 查找连通组件
        contours_all, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours_all:
            area = cv2.contourArea(contour)
            if area > 5:  # 过滤掉非常小的区域
                # 获取边界矩形
                x, y, w, h = cv2.boundingRect(contour)
                
                # 判断是否为水平或垂直线段
                aspect_ratio = max(w, h) / min(w, h)
                
                # 如果宽高比大于某个阈值（比如3），认为是线段
                if aspect_ratio > 3:
                    # 水平线：宽度 > 高度
                    if w > h:
                        line_widths.append(h)  # 水平线的宽度是高度
                    else:  # 垂直线：高度 > 宽度
                        line_widths.append(w)  # 垂直线的宽度是宽度
                elif area > 50:  # 对于面积较大的区域，检查是否可能是线段
                    # 再次检查宽高比
                    if aspect_ratio > 2:
                        if w > h:
                            line_widths.append(h)
                        else:
                            line_widths.append(w)
    
    # 对线段宽度进行聚类
    if len(line_widths) > 0:
        # 使用K-means对线宽进行聚类
        X = np.array(line_widths).reshape(-1, 1)
        n_clusters = min(len(set(line_widths)), 5)  # 最多5个聚类
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # 找到包含最多线段的聚类
            unique_labels, counts = np.unique(labels, return_counts=True)
            dominant_cluster_idx = unique_labels[np.argmax(counts)]
            
            # 计算主导聚类的平均宽度
            dominant_widths = X[labels == dominant_cluster_idx].flatten()
            avg_dominant_width = np.mean(dominant_widths)
        else:
            avg_dominant_width = np.mean(line_widths)
    else:
        avg_dominant_width = 0
    
    print(f"Average dominant line width: {avg_dominant_width:.2f}")
    print(f"Number of horizontal/vertical line segments detected: {len(line_widths)}")
    
    # 7. 创建经过线宽校验的图像
    print("Creating validated rectangles based on line width...")
    
    # 创建带验证结果的图像
    validated_img = img.copy()
    for i, region in enumerate(clustered_regions):
        x, y, width, height = region['bbox']
        
        # 检查最短边是否接近平均线宽
        min_dimension = min(width, height)
        
        # 判断是否与平均线宽接近（允许一定误差范围）
        # 修改：容差放宽至统计值的2倍
        tolerance = avg_dominant_width * 2.0  # 放宽至2倍容差
        if abs(min_dimension - avg_dominant_width) <= tolerance and avg_dominant_width > 0:
            # 最短边接近平均线宽，将矩形内的白色区域变成黑色
            # 获取ROI区域
            roi = img[y:y+height, x:x+width]
            
            # 将ROI中的白色区域（灰度值 >= white_threshold）变成黑色
            roi_gray = img_gray[y:y+height, x:x+width]
            white_in_roi = roi_gray >= white_threshold
            roi[white_in_roi] = [0, 0, 0]  # 变成黑色
            
            # 将处理后的ROI放回原图像
            validated_img[y:y+height, x:x+width] = roi
        else:
            # 不符合条件，将矩形内的白色区域变成红色
            # 获取ROI区域
            roi = img[y:y+height, x:x+width]
            
            # 将ROI中的白色区域（灰度值 >= white_threshold）变成红色
            roi_gray = img_gray[y:y+height, x:x+width]
            white_in_roi = roi_gray >= white_threshold
            roi[white_in_roi] = [0, 0, 255]  # 变成红色
            
            # 将处理后的ROI放回原图像
            validated_img[y:y+height, x:x+width] = roi
        
        # 在矩形边上添加长宽尺寸标注
        # 标注在矩形下方
        cv2.putText(validated_img, f'{width}x{height}', 
                   (x, y + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # 同时在矩形上方也标注（避免被遮挡）
        if y > 15:  # 确保文本不会超出图像顶部
            cv2.putText(validated_img, f'{width}x{height}', 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # 8. 绘制六个结果图
    print("Creating visualization...")
    
    # 创建六个子图 (2x3)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 子图1: 原始图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('1. Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 子图2: 灰度直方图
    axes[0, 1].plot(hist, color='black', linewidth=1.5, label='Original histogram')
    
    # 绘制平滑后的直方图用于波谷检测
    hist_smooth = gaussian_filter1d(hist_ravel, sigma=2)
    axes[0, 1].plot(hist_smooth, color='red', linewidth=1.5, alpha=0.7, label='Smoothed histogram')
    
    axes[0, 1].set_xlabel('Pixel Intensity', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('2. Grayscale Histogram with Valley Detection', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 标记背景峰值和白色阈值
    axes[0, 1].axvline(x=bg_gray, color='red', linestyle='--', linewidth=2, 
                   label=f'Background peak: {bg_gray}')
    axes[0, 1].axvline(x=white_threshold, color='blue', linestyle='--', linewidth=2, 
                   label=f'White threshold: {white_threshold}')
    
    # 标记波谷
    # 在背景峰右侧寻找波谷
    search_start = int(bg_gray)
    search_range = hist_smooth[search_start:]
    diff = np.diff(search_range)
    sign_change = np.where(np.diff(np.sign(diff)) > 0)[0]
    
    if len(sign_change) > 0:
        first_valley = search_start + sign_change[0]
        axes[0, 1].axvline(x=first_valley, color='green', linestyle='--', linewidth=2, 
                       label=f'First valley: {first_valley}')
        axes[0, 1].plot(first_valley, hist_smooth[first_valley], 'go', markersize=8, label='Detected valley')
    
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].set_xlim([0, 255])
    
    # 子图3: 白色区域掩码（白色区域为255，黑色区域为0）
    axes[0, 2].imshow(white_mask, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f'3. White Regions Mask ({len(regions_info)} regions)', 
                     fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 添加统计信息
    white_pixel_count = np.sum(white_mask > 0)
    white_percentage = (white_pixel_count / total_pixels) * 100
    axes[0, 2].text(0.02, 0.98, 
                f'White pixels: {white_pixel_count:,} ({white_percentage:.1f}%)',
                transform=axes[0, 2].transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 添加区域数量信息
    axes[0, 2].text(0.02, 0.92, 
                f'Regions detected: {len(regions_info)}',
                transform=axes[0, 2].transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 子图4: 聚类结果及矩形框标注（已过滤接近整图的矩形）
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title(f'4. Clustering Result ({len(clustered_regions)} clusters, filtered)', 
                        fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # 在原始图像上绘制聚类后的边界框（已过滤）
    colors = plt.cm.get_cmap('tab20', len(clustered_regions))
    for i, region in enumerate(clustered_regions):
        x, y, width, height = region['bbox']
        
        # 随机选择颜色绘制矩形框
        color = colors(i)[:3]  # 获取RGB值
        rect = plt.Rectangle((x, y), width, height, fill=False, 
                           edgecolor=color, linewidth=2)
        axes[1, 0].add_patch(rect)
        
        # 添加标签
        axes[1, 0].text(x, y-5, f'{i+1}', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # 添加聚类统计信息
    axes[1, 0].text(0.02, 0.98, 
                f'Total clusters: {len(clustered_regions)}',
                transform=axes[1, 0].transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 子图5: 经过线宽校验的矩形（已添加尺寸标注）
    validated_img_rgb = cv2.cvtColor(validated_img, cv2.COLOR_BGR2RGB)
    axes[1, 1].imshow(validated_img_rgb)
    axes[1, 1].set_title(f'5. Width-Validated Rectangles\n(Avg line width: {avg_dominant_width:.2f}px)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 添加说明文本
    if avg_dominant_width > 0:
        axes[1, 1].text(0.02, 0.98, 
                    'Black: Close to avg line width\nRed: Different from avg line width',
                    transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 子图6: 水平和垂直黑色线段宽度统计
    axes[1, 2].set_title('6. Horizontal & Vertical Black Line Widths', fontsize=14, fontweight='bold')
    
    if len(line_widths) > 0:
        # 绘制线宽分布直方图
        n, bins, patches = axes[1, 2].hist(line_widths, bins=min(20, len(set(line_widths))), 
                                         color='blue', alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(avg_dominant_width, color='red', linestyle='--', linewidth=2, 
                          label=f'Avg dominant width: {avg_dominant_width:.2f}')
        axes[1, 2].set_xlabel('Line Width (pixels)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title(f'Horizontal & Vertical Line Widths\n({len(line_widths)} segments detected)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f'Statistics:\nTotal lines: {len(line_widths)}\nMean: {np.mean(line_widths):.2f}\nStd: {np.std(line_widths):.2f}\nMin: {np.min(line_widths):.2f}\nMax: {np.max(line_widths):.2f}'
        axes[1, 2].text(0.02, 0.98, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1, 2].text(0.5, 0.5, 'No horizontal/vertical\nline segments detected', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = "white_regions_detection_complete.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # 保存白色掩码图像（单独保存二值图像）
    white_mask_output_path = "white_mask_binary.png"
    cv2.imwrite(white_mask_output_path, white_mask)
    print(f"White mask saved to: {white_mask_output_path}")
    
    # 保存聚类结果图像
    clustered_result_path = "clustered_regions.png"
    result_img = img.copy()
    
    # 为每个聚类绘制边界框（已过滤）
    for i, region in enumerate(clustered_regions):
        x, y, width, height = region['bbox']
        # 随机颜色（使用固定种子确保每次运行颜色一致）
        color_idx = i % 10
        colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                      (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0), 
                      (139, 69, 19), (255, 192, 203)]
        color = colors_list[color_idx]
        
        cv2.rectangle(result_img, (x, y), (x + width, y + height), color, 2)
        cv2.putText(result_img, str(i+1), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1)
    
    cv2.imwrite(clustered_result_path, result_img)
    print(f"Clustered regions image saved to: {clustered_result_path}")
    
    # 保存经过线宽校验的图像
    validated_result_path = "validated_regions.png"
    cv2.imwrite(validated_result_path, validated_img)
    print(f"Width-validated regions image saved to: {validated_result_path}")
    
    # 6. 输出结果
    print("\n" + "=" * 60)
    print("WHITE REGIONS DETECTION RESULTS")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Size: {w}x{h} pixels")
    print(f"Background peak gray level: {bg_gray}")
    print(f"White threshold (auto-determined): {white_threshold}")
    print(f"White pixels: {white_pixel_count:,} ({white_percentage:.2f}%)")
    print(f"Contour-based regions detected: {len(regions_info)}")
    print(f"Clustering-based regions detected: {len(clustered_regions)}")
    print(f"Average dominant line width: {avg_dominant_width:.2f} pixels")
    print(f"Number of horizontal/vertical line segments analyzed: {len(line_widths)}")
    print(f"Minimum area threshold: {min_area}")
    print(f"Output saved to: {output_path}")
    print(f"White mask saved to: {white_mask_output_path}")
    print(f"Clustered result saved to: {clustered_result_path}")
    print(f"Validated result saved to: {validated_result_path}")
    
    # 打印聚类后区域详细信息
    if len(clustered_regions) > 0:
        print("\nClustered White Regions (after filtering):")
        print("-" * 50)
        for region in clustered_regions[:5]:  # 只显示前5个区域
            bbox = region['bbox']
            min_dim = min(bbox[2], bbox[3])
            tolerance = avg_dominant_width * 2.0  # 放宽至2倍容差
            matches_line_width = abs(min_dim - avg_dominant_width) <= tolerance if avg_dominant_width > 0 else False
            
            print(f"Region {region['id']}:")
            print(f"  Position: ({bbox[0]}, {bbox[1]})")
            print(f"  Size: {bbox[2]}x{bbox[3]} pixels")
            print(f"  Area: {region['area']:,} pixels")
            print(f"  Min dimension: {min_dim} pixels")
            print(f"  Matches line width: {matches_line_width}")
            print(f"  Aspect ratio: {region['aspect_ratio']:.2f}")
            print()
        
        if len(clustered_regions) > 5:
            print(f"... and {len(clustered_regions) - 5} more regions")
    
    # 返回结果
    return {
        'original_image': img,
        'white_mask': white_mask,
        'clustered_mask': clustered_mask,
        'regions_info': regions_info,
        'clustered_regions': clustered_regions,
        'line_widths': line_widths,
        'avg_dominant_width': avg_dominant_width,
        'bg_gray_level': bg_gray,
        'white_threshold': white_threshold,
        'white_pixel_count': white_pixel_count,
        'white_percentage': white_percentage,
        'output_path': output_path,
        'white_mask_path': white_mask_output_path,
        'clustered_result_path': clustered_result_path,
        'validated_result_path': validated_result_path
    }

# 主函数
if __name__ == "__main__":
    # 指定输入图像路径
    image_path = "pcb_chip_removal_single.png"
    
    # 可以调整的参数：
    # white_threshold: 白色阈值，默认为None（自动确定）
    # min_area_ratio: 最小面积比例，默认0.0005（图像面积的0.05%），用于过滤小区域
    
    try:
        print("=" * 60)
        print("WHITE REGIONS DETECTION WITH CLUSTERING AND VALIDATION")
        print("=" * 60)
        print("Note: White threshold will be automatically determined as the first valley")
        print("      after the background peak in the histogram.")
        
        # 运行检测
        results = detect_and_mark_white_regions(
            image_path=image_path,
            white_threshold=None,  # 设为None以自动确定阈值
            min_area_ratio=0.0005  # 可根据需要调整
        )
        
        print("\n✅ White regions detection with clustering and validation completed successfully!")
        print(f"\nSummary:")
        print(f"  - Background peak at: {results['bg_gray_level']}")
        print(f"  - Auto-determined white threshold: {results['white_threshold']}")
        print(f"  - Contour-based regions: {len(results['regions_info'])}")
        print(f"  - Clustered regions (filtered): {len(results['clustered_regions'])}")
        print(f"  - Average dominant line width: {results['avg_dominant_width']:.2f} pixels")
        print(f"  - Number of horizontal/vertical line segments: {len(results['line_widths'])}")
        print(f"  - White pixels: {results['white_pixel_count']:,} ({results['white_percentage']:.1f}%)")
        print(f"  - Output saved to: {results['output_path']}")
        print(f"  - White mask saved to: {results['white_mask_path']}")
        print(f"  - Clustered result saved to: {results['clustered_result_path']}")
        print(f"  - Validated result saved to: {results['validated_result_path']}")
        
        # 提供调整建议
        if len(results['clustered_regions']) == 0:
            print("\n⚠️  No regions detected. You may try:")
            print(f"   1. Lowering the white threshold manually (e.g., white_threshold={max(0, results['bg_gray_level']-20)})")
            print(f"   2. Decreasing min_area_ratio (e.g., min_area_ratio=0.0001)")
        elif len(results['clustered_regions']) > 50:
            print("\n⚠️  Many regions detected. You may try:")
            print(f"   1. Increasing the white threshold manually (e.g., white_threshold={min(255, results['white_threshold']+20)})")
            print(f"   2. Increasing min_area_ratio (e.g., min_area_ratio=0.001)")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()