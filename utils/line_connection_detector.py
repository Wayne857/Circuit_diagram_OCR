import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path
import json
from collections import Counter


class LineConnectionDetector:
    """线连接检测器，用于检测PCB图像中的导线连接关系"""

    def __init__(self):
        pass

    def zhang_suen_skeletonize(self, binary):
        """Zhang-Suen 骨架提取算法"""
        img = binary.copy()
        h, w = img.shape
        neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        
        while True:
            delete_pixels = []
            # 第一步删除
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if img[y, x] == 255:
                        count = sum(img[y+dy, x+dx] == 255 for dy, dx in neighbors)
                        if not (2 <= count <= 6):
                            continue
                        transitions = 0
                        for i in range(8):
                            curr = img[y+neighbors[i][0], x+neighbors[i][1]]
                            next_ = img[y+neighbors[(i+1)%8][0], x+neighbors[(i+1)%8][1]]
                            if curr == 0 and next_ == 255:
                                transitions += 1
                        if transitions != 1:
                            continue
                        p2 = img[y-1, x] == 255
                        p4 = img[y, x+1] == 255
                        p6 = img[y+1, x] == 255
                        p8 = img[y, x-1] == 255
                        if not (p2 and p4 and p6) and not (p4 and p6 and p8):
                            delete_pixels.append((y, x))
            for y, x in delete_pixels:
                img[y, x] = 0
            
            # 第二步删除
            delete_pixels = []
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if img[y, x] == 255:
                        count = sum(img[y+dy, x+dx] == 255 for dy, dx in neighbors)
                        if not (2 <= count <= 6):
                            continue
                        transitions = 0
                        for i in range(8):
                            curr = img[y+neighbors[i][0], x+neighbors[i][1]]
                            next_ = img[y+neighbors[(i+1)%8][0], x+neighbors[(i+1)%8][1]]
                            if curr == 0 and next_ == 255:
                                transitions += 1
                        if transitions != 1:
                            continue
                        p2 = img[y-1, x] == 255
                        p4 = img[y, x+1] == 255
                        p6 = img[y+1, x] == 255
                        p8 = img[y, x-1] == 255
                        if not (p2 and p4 and p8) and not (p2 and p6 and p8):
                            delete_pixels.append((y, x))
            for y, x in delete_pixels:
                img[y, x] = 0
            
            if len(delete_pixels) == 0:
                break
        
        return img

    def distance(self, p1, p2):
        """计算两个点之间的欧氏距离"""
        return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def calculate_segment_width_statistics(self, segment_path, original_image):
        """
        计算线段的宽度统计信息
        使用垂直扫描的方法统计线段的实际像素宽度
        Args:
            segment_path: 线段的骨架路径点列表
            original_image: 原始图像
        Returns:
            线段的宽度统计信息字典
        """
        if len(original_image.shape) == 3:
            gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = original_image
        
        # 二值化图像
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        widths = []
        
        # 对路径上的关键点进行宽度测量
        sample_points = segment_path[::max(1, len(segment_path)//10)]  # 采样10个点
        if len(segment_path) < 10:
            sample_points = segment_path
        
        for point in sample_points:
            x, y = point
            if 0 <= x < w and 0 <= y < h:
                # 计算该点的垂直宽度
                width = self._calculate_point_width(binary, x, y)
                if width > 0:
                    widths.append(width)
        
        if not widths:
            return {'avg_width': 1.0, 'max_width': 1.0, 'min_width': 1.0, 'width_list': [1.0]}
        
        # 统计信息
        width_counter = Counter(widths)
        most_common_width = width_counter.most_common(1)[0][0]
        
        return {
            'avg_width': float(np.mean(widths)),
            'max_width': float(np.max(widths)),
            'min_width': float(np.min(widths)),
            'most_common_width': float(most_common_width),
            'width_list': [float(w) for w in widths]
        }
    
    def _calculate_point_width(self, binary, x, y):
        """
        计算单个点的垂直宽度
        Args:
            binary: 二值化图像
            x, y: 点坐标
        Returns:
            该点的垂直宽度
        """
        h, w = binary.shape
        
        # 首先确认这个点是否在线段上
        if binary[y, x] != 255:
            return 1
        
        # 向上搜索（限制在较小范围内）
        up_count = 0
        for dy in range(1, min(15, y + 1)):  # 最多搜索15像素
            if binary[y - dy, x] == 255:
                up_count += 1
            else:
                break
        
        # 向下搜索
        down_count = 0
        for dy in range(1, min(15, h - y)):
            if binary[y + dy, x] == 255:
                down_count += 1
            else:
                break
        
        # 向左搜索
        left_count = 0
        for dx in range(1, min(15, x + 1)):
            if binary[y, x - dx] == 255:
                left_count += 1
            else:
                break
        
        # 向右搜索
        right_count = 0
        for dx in range(1, min(15, w - x)):
            if binary[y, x + dx] == 255:
                right_count += 1
            else:
                break
        
        # 取垂直方向和水平方向的最大值作为宽度
        vertical_width = up_count + down_count + 1
        horizontal_width = left_count + right_count + 1
        
        # 返回较小的值，避免过度估计
        return min(vertical_width, horizontal_width, 10)
    
    def filter_segments_by_width_statistics(self, segments, width_stats_list, threshold_ratio=2.0, show_plot=False):
        """
        基于宽度统计信息过滤线段
        Args:
            segments: 线段列表
            width_stats_list: 对应的宽度统计信息列表
            threshold_ratio: 宽度比例阈值，小于该比例的线段将被过滤
            show_plot: 是否显示宽度分布图
        Returns:
            过滤后的线段列表
        """
        if len(segments) <= 1:
            return segments
        
        # 提取平均宽度
        avg_widths = [stats['avg_width'] for stats in width_stats_list]
        
        # 找到最大宽度
        max_width = max(avg_widths)
        threshold = max_width / threshold_ratio
        
        # 显示统计图
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            segment_nums = range(1, len(avg_widths) + 1)
            colors = ['red' if w < threshold else 'green' for w in avg_widths]
            bars = plt.bar(segment_nums, avg_widths, color=colors)
            plt.axhline(y=threshold, color='orange', linestyle='--', label='阈值 ({:.2f})'.format(threshold))
            plt.xlabel('线段编号')
            plt.ylabel('宽度')
            plt.title('线段宽度分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 在柱状图上添加数值标签
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        '{:.2f}'.format(avg_widths[i]),
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
        
        # 过滤线段
        filtered_segments = []
        filtered_stats = []
        
        for i, (segment, stats) in enumerate(zip(segments, width_stats_list)):
            avg_width = stats['avg_width']
            # 如果宽度大于等于阈值，则保留
            if avg_width >= threshold:
                filtered_segments.append(segment)
                filtered_stats.append(stats)
                print(f"  保留线段 {i+1}，宽度: {avg_width:.2f}")
            else:
                print(f"  过滤线段 {i+1}，宽度: {avg_width:.2f} (阈值: {threshold:.2f})")
        
        return filtered_segments, filtered_stats
    
    def _point_in_polygon(self, x, y, polygon):
        """
        使用射线投射算法检查点是否在多边形内
        polygon: 多边形顶点列表，格式为[[x1,y1], [x2,y2], ...]
        """
        if not polygon:
            return False
        
        n = len(polygon)
        inside = False
        
        # 将第一个顶点存储为起始点
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            # 获取下一个顶点
            p2x, p2y = polygon[i % n]
            
            # 检查点是否在当前边的y范围内
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                # 检查射线是否与当前边相交
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    
                # 如果点在边上或射线穿过边，则翻转inside标志
                if p1x == p2x or x <= xinters:
                    inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside

    def merge_close_points(self, points, threshold=3):
        """合并距离小于阈值的特征点（去重）"""
        merged = []
        used = [False] * len(points)
        for i in range(len(points)):
            if used[i]:
                continue
            curr_type, curr_point = points[i]
            # 找所有和当前点距离小于阈值的点
            cluster = [curr_point]
            for j in range(i+1, len(points)):
                if not used[j] and self.distance(curr_point, points[j][1]) < threshold:
                    cluster.append(points[j][1])
                    used[j] = True
            # 取聚类中心作为合并后的点
            avg_x = int(sum(p[0] for p in cluster)/len(cluster))
            avg_y = int(sum(p[1] for p in cluster)/len(cluster))
            merged.append((curr_type, (avg_x, avg_y)))
            used[i] = True
        return merged

    def merge_close_segments(self, segments, threshold=5):
        """合并端点距离小于阈值的线段（避免一条线拆成多条）
        但保留有意义的转角（如L型、Z型结构）
        """
        merged_segments = []
        used = [False] * len(segments)
        
        for i in range(len(segments)):
            if used[i]:
                continue
            start_i, end_i, path_i = segments[i]
            current_segment = (start_i, end_i, path_i)
            
            # 寻找可以合并的线段
            while True:
                merged = False
                for j in range(len(segments)):
                    if i == j or used[j]:
                        continue
                    start_j, end_j, path_j = segments[j]
                    
                    # 检查当前线段的端点是否和j线段的端点接近
                    # 包括所有可能的端点配对情况
                    merge_condition = (self.distance(end_i, start_j) < threshold or 
                                     self.distance(end_i, end_j) < threshold or
                                     self.distance(start_i, start_j) < threshold or
                                     self.distance(start_i, end_j) < threshold)
                    
                    # 如果满足合并条件，还需要检查是否应该保留转角
                    if merge_condition:
                        should_merge = self._should_merge_segments(
                            current_segment, (start_j, end_j, path_j), threshold)
                        
                        if should_merge:
                            # 合并两条线段的路径（处理所有四种连接情况）
                            if self.distance(end_i, start_j) < threshold:
                                # end_i -> start_j: 正常连接
                                new_path = path_i + path_j
                                new_start = start_i
                                new_end = end_j
                            elif self.distance(end_i, end_j) < threshold:
                                # end_i -> end_j: 反向连接
                                new_path = path_i + path_j[::-1]
                                new_start = start_i
                                new_end = start_j
                            elif self.distance(start_i, start_j) < threshold:
                                # start_i -> start_j: 反向连接
                                new_path = path_j + path_i
                                new_start = start_j
                                new_end = end_i
                            else: # distance(start_i, end_j) < threshold
                                # start_i -> end_j: 反向连接
                                new_path = path_j[::-1] + path_i
                                new_start = end_j
                                new_end = end_i
                            
                            current_segment = (new_start, new_end, new_path)
                            used[j] = True
                            merged = True
                            break
                if not merged:
                    break
            
            merged_segments.append(current_segment)
            used[i] = True
        
        return merged_segments
    
    def _should_merge_segments(self, segment1, segment2, threshold):
        """
        判断两个线段是否应该合并
        对于直角和Z字型结构，允许合并成连通线段
        Args:
            segment1: 第一个线段 (start, end, path)
            segment2: 第二个线段 (start, end, path)
            threshold: 合并阈值
        Returns:
            bool: 是否应该合并
        """
        start1, end1, path1 = segment1
        start2, end2, path2 = segment2
        
        # 计算所有可能的端点距离
        distances = [
            self.distance(end1, start2),    # segment1终点 -> segment2起点
            self.distance(end1, end2),      # segment1终点 -> segment2终点
            self.distance(start1, start2),  # segment1起点 -> segment2起点
            self.distance(start1, end2)     # segment1起点 -> segment2终点
        ]
        min_distance = min(distances)
        
        # 计算两个线段的方向向量
        def get_direction_vector(start, end):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = max(1, self.distance(start, end))
            return (dx/length, dy/length)
        
        # 获取线段方向
        dir1 = get_direction_vector(start1, end1)
        dir2 = get_direction_vector(start2, end2)
        
        # 计算方向向量的夹角（点积）
        dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
        angle_cos = abs(dot_product)  # 取绝对值，因为方向可能相反
        
        # 改进的合并策略：
        # 1. 方向相似的线段（接近平行）- 合并
        # 2. 方向差异较大的线段（接近垂直）- 也允许合并（用于直角/Z字型）
        # 3. 只有完全相反方向且连接松散的线段才不合并
        
        # 更宽松的角度阈值
        angle_threshold = 0.3  # cos(72.5°) ≈ 0.3，允许更大角度差异的合并
        
        if angle_cos >= angle_threshold:
            # 方向相似或中等差异，可以合并
            return True
        else:
            # 即使方向差异较大，如果连接很紧密也允许合并
            # 这样可以处理直角和Z字型结构
            close_connection_threshold = threshold * 0.5
            if min_distance <= close_connection_threshold:
                return True
            return False

    def preprocess_image(self, image_path):
        """读取图片并预处理（新增闭运算填补微小断裂）"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图片，请检查路径：{image_path}")
        
        # 二值化
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # 去噪+闭运算（填补1-2像素的微小断裂）
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # 开运算去噪
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) # 闭运算补断裂
        
        # 骨架提取
        skeleton = self.zhang_suen_skeletonize(binary)
        
        # 骨架后再做一次闭运算，确保连续
        skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, np.ones((1,1), np.uint8))
        return img, binary, skeleton

    def detect_feature_points(self, skeleton):
        """检测特征点（优化邻域判断，增加容错）"""
        h, w = skeleton.shape
        feature_points = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 255:
                    # 计算8邻域有效像素数（排除自身）
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2] == 255) - 1
                    # 放宽判断：端点（neighbors=1），交点（neighbors>=3），避免误判
                    if neighbors == 1:
                        feature_points.append(('endpoint', (x, y)))
                    elif neighbors >= 3:
                        feature_points.append(('junction', (x, y)))
        # 合并距离极近的特征点
        feature_points = self.merge_close_points(feature_points, threshold=3)
        return feature_points

    def trace_wire_segments(self, skeleton, feature_points):
        """追踪导线线段（重写的追踪算法，专门处理直角和Z型结构）"""
        h, w = skeleton.shape
        visited = set()
        segments = []
        point_coords = {p[1] for p in feature_points}

        def trace_from_point(start_point, target_point):
            """从起点追踪到目标点，遵循直线路径"""
            if start_point == target_point:
                return [start_point]
            
            path = [start_point]
            current = start_point
            visited.add(current)
            
            # 计算主要追踪方向
            dx = target_point[0] - start_point[0]
            dy = target_point[1] - start_point[1]
            
            # 确定主要追踪方向
            if abs(dx) > abs(dy):  # 主要水平方向
                step_x = 1 if dx > 0 else -1
                step_y = 0
            else:  # 主要垂直方向
                step_x = 0
                step_y = 1 if dy > 0 else -1
            
            # 追踪到目标点或遇到障碍
            while current != target_point:
                x, y = current
                # 尝试主要方向
                next_x, next_y = x + step_x, y + step_y
                
                if (0 <= next_x < w and 0 <= next_y < h and 
                    skeleton[next_y, next_x] == 255 and 
                    (next_x, next_y) not in visited):
                    current = (next_x, next_y)
                    visited.add(current)
                    path.append(current)
                    continue
                
                # 如果主要方向不行，尝试其他方向
                found = False
                for dx_offset in [-1, 0, 1]:
                    for dy_offset in [-1, 0, 1]:
                        if dx_offset == 0 and dy_offset == 0:
                            continue
                        next_x, next_y = x + dx_offset, y + dy_offset
                        if (0 <= next_x < w and 0 <= next_y < h and 
                            skeleton[next_y, next_x] == 255 and 
                            (next_x, next_y) not in visited):
                            current = (next_x, next_y)
                            visited.add(current)
                            path.append(current)
                            found = True
                            break
                    if found:
                        break
                
                if not found or current == target_point:
                    break
            
            return path

        # 为每个特征点找到最近的相邻特征点
        def find_nearest_neighbors(point, all_points):
            """找到最近的相邻特征点"""
            neighbors = []
            px, py = point
            
            for other_point in all_points:
                if other_point != point:
                    distance = self.distance(point, other_point)
                    if distance < 50:  # 限制搜索范围
                        neighbors.append((distance, other_point))
            
            # 按距离排序
            neighbors.sort(key=lambda x: x[0])
            return [point for _, point in neighbors[:2]]  # 返回最近的2个点

        # 追踪所有特征点
        feature_coords = [p[1] for p in feature_points]
        for point_type, point in feature_points:
            if point not in visited:
                # 找到最近的相邻点
                neighbors = find_nearest_neighbors(point, feature_coords)
                for neighbor in neighbors:
                    if neighbor not in visited or neighbor in point_coords:
                        path = trace_from_point(point, neighbor)
                        if len(path) >= 2:
                            segments.append((path[0], path[-1], path))
        
        # 合并距离极近的线段
        segments = self.merge_close_segments(segments, threshold=5)
        return segments

    def build_connection_graph(self, segments):
        """构建连接关系图"""
        G = nx.Graph()
        for start, end, _ in segments:
            G.add_edge(start, end)
        return G

    def visualize_results(self, img, skeleton, feature_points, segments, output_dir, image_filename):
        """保存可视化结果到指定目录"""
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存骨架图
        skeleton_output_path = output_path / f"{image_filename}_skeleton.jpg"
        cv2.imwrite(str(skeleton_output_path), skeleton)
        
        # 保存带有线段和特征点的图像
        result_img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        
        # 绘制特征点
        for point_type, (x, y) in feature_points:
            color = (0, 0, 255) if point_type == 'endpoint' else (255, 0, 0)  # 红色端点，蓝色交点
            cv2.circle(result_img, (x, y), 3, color, -1)
        
        # 绘制线段并添加序号
        for idx, (start, end, path) in enumerate(segments, 1):
            # 绘制导线线段
            for i in range(len(path) - 1):
                cv2.line(result_img, path[i], path[i+1], (0, 255, 0), 1)
            
            # 计算线段中点（序号显示位置）
            mid_idx = len(path) // 2
            mid_x, mid_y = path[mid_idx]
            
            # 绘制序号
            cv2.putText(result_img, str(idx), (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)  # 黄色序号
        
        # 保存结果图像
        result_output_path = output_path / f"{image_filename}_with_segments.jpg"
        cv2.imwrite(str(result_output_path), result_img)
        
        return skeleton_output_path, result_output_path

    def find_connected_components_or_texts(self, segments, components, texts, radius_threshold=30):
        """
        查找线段两端连接的元件或文本
        """
        connections = []
        for idx, (start_point, end_point, path) in enumerate(segments):
            connection_info = {
                'segment_id': idx + 1,  # 线段编号（从1开始，与可视化中的编号一致）
                'segment': (start_point, end_point, path),
                'start_connected': self._find_nearest_object(start_point, components, texts, radius_threshold),
                'end_connected': self._find_nearest_object(end_point, components, texts, radius_threshold)
            }
            connections.append(connection_info)
        return connections

    def _find_nearest_object(self, point, components, texts, radius_threshold):
        """
        在指定半径内查找最近的元件或文本
        优先检查点是否在元件的分割掩码内
        排除line类别的连接
        """
        px, py = point
        nearest_obj = None
        min_dist = float('inf')
        
        # 首先检查点是否在任何元件的分割掩码内
        for comp in components:
            # 排除line类别
            if comp['category'] == 'line':
                continue
            # 检查是否有分割掩码信息
            if 'segmentation' in comp and comp['segmentation']:
                # 创建掩码并检查点是否在其中
                segmentation_coords = comp['segmentation']
                if self._point_in_polygon(px, py, segmentation_coords):
                    # 如果点在元件的分割区域内，直接返回该元件
                    return {
                        'type': 'component',
                        'data': comp,
                        'distance': 0  # 距离为0，因为点在内部
                    }
        
        # 然后检查文本的分割掩码
        for text in texts:
            # 检查是否有分割掩码信息
            if 'segmentation' in text and text['segmentation']:
                # 创建掩码并检查点是否在其中
                segmentation_coords = text['segmentation']
                if self._point_in_polygon(px, py, segmentation_coords):
                    # 如果点在文本的分割区域内，直接返回该文本
                    return {
                        'type': 'text',
                        'data': text,
                        'distance': 0  # 距离为0，因为点在内部
                    }
        
        # 如果点不在任何分割掩码内，使用原有的距离检查
        # 检查元件
        for comp in components:
            # 排除line类别
            if comp['category'] == 'line':
                continue
            cx1, cy1, cx2, cy2 = comp['bbox']
            comp_center_x = (cx1 + cx2) / 2
            comp_center_y = (cy1 + cy2) / 2
            dist = self.distance((px, py), (comp_center_x, comp_center_y))
            if dist < min_dist and dist <= radius_threshold:
                min_dist = dist
                nearest_obj = {
                    'type': 'component',
                    'data': comp,
                    'distance': dist
                }
        
        # 检查文本
        for text in texts:
            tx1, ty1, tx2, ty2 = text['coord']
            text_center_x = (tx1 + tx2) / 2
            text_center_y = (ty1 + ty2) / 2
            dist = self.distance((px, py), (text_center_x, text_center_y))
            if dist < min_dist and dist <= radius_threshold:
                min_dist = dist
                nearest_obj = {
                    'type': 'text',
                    'data': text,
                    'distance': dist
                }
        
        return nearest_obj

    def generate_detailed_visualizations(self, connections, original_image, output_dir, image_filename):
        """
        为每条线段生成详细的连接关系可视化图像
        """
        # 创建details目录
        details_dir = Path(output_dir) / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每条连接生成详细图像
        for connection in connections:
            segment_id = connection['segment_id']
            start_point, end_point, path = connection['segment']
            start_connected = connection['start_connected']
            end_connected = connection['end_connected']
            
            # 复制原始图像用于绘制
            result_img = original_image.copy()
            
            # 绘制线段（使用绿色，线宽为2以突出显示）
            for i in range(len(path) - 1):
                cv2.line(result_img, path[i], path[i+1], (0, 255, 0), 2)
            
            # 绘制线段编号
            mid_idx = len(path) // 2
            if 0 <= mid_idx < len(path):
                mid_point = path[mid_idx]
                cv2.putText(result_img, str(segment_id), mid_point, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # 黄色编号
            
            # 高亮起点连接的对象
            if start_connected:
                if start_connected['type'] == 'component':
                    comp_data = start_connected['data']
                    x1, y1, x2, y2 = comp_data['bbox']
                    # 用蓝色矩形框高亮元件
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色框
                    # 标注元件类别
                    cv2.putText(result_img, f"{comp_data['category']}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                elif start_connected['type'] == 'text':
                    text_data = start_connected['data']
                    x1, y1, x2, y2 = text_data['coord']
                    # 用红色矩形框高亮文本
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框
                    # 标注文本内容
                    cv2.putText(result_img, f"文本:{text_data['text']}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 高亮终点连接的对象
            if end_connected:
                if end_connected['type'] == 'component':
                    comp_data = end_connected['data']
                    x1, y1, x2, y2 = comp_data['bbox']
                    # 用蓝色矩形框高亮元件
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色框
                    # 标注元件类别
                    cv2.putText(result_img, f"{comp_data['category']}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                elif end_connected['type'] == 'text':
                    text_data = end_connected['data']
                    x1, y1, x2, y2 = text_data['coord']
                    # 用红色矩形框高亮文本
                    cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色框
                    # 标注文本内容
                    cv2.putText(result_img, f"文本:{text_data['text']}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # 保存详细图像
            detail_path = details_dir / f"connection_{segment_id}.jpg"
            cv2.imwrite(str(detail_path), result_img)
            
            print(f"  保存连接 {segment_id} 的详细图像: {detail_path}")
    
    def generate_connection_markdown(self, connections, output_dir, image_filename):
        """
        生成连接关系的文本文件
        """
        markdown_path = Path(output_dir) / "details" / "connection_summary.md"
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(f"# 线段连接关系摘要\n\n")
            f.write(f"## 图像: {image_filename}\n\n")
            f.write(f"总共检测到 {len(connections)} 个连接关系\n\n")
            
            for connection in connections:
                segment_id = connection['segment_id']
                start_connected = connection['start_connected']
                end_connected = connection['end_connected']
                
                f.write(f"### 连接 {segment_id}\n")
                
                # 描述起点连接的对象
                start_desc = "无连接"
                if start_connected:
                    if start_connected['type'] == 'component':
                        comp_data = start_connected['data']
                        start_desc = f"元件: {comp_data['category']}, 坐标: {comp_data['bbox']}"
                    elif start_connected['type'] == 'text':
                        text_data = start_connected['data']
                        start_desc = f"文本: '{text_data['text']}', 坐标: {text_data['coord']}"
                
                # 描述终点连接的对象
                end_desc = "无连接"
                if end_connected:
                    if end_connected['type'] == 'component':
                        comp_data = end_connected['data']
                        end_desc = f"元件: {comp_data['category']}, 坐标: {comp_data['bbox']}"
                    elif end_connected['type'] == 'text':
                        text_data = end_connected['data']
                        end_desc = f"文本: '{text_data['text']}', 坐标: {text_data['coord']}"
                
                f.write(f"- 起点连接: {start_desc}\n")
                f.write(f"- 终点连接: {end_desc}\n")
                f.write("\n")
        
        print(f"  保存连接关系摘要: {markdown_path}")
    
    def detect_line_connections(self, image_path, output_dir, image_filename, components=None, texts=None, original_image=None):
        """检测线连接关系的主函数"""
        try:
            # 1. 预处理（含闭运算补断裂）
            img, binary, skeleton = self.preprocess_image(image_path)
            
            # 2. 检测特征点（含去重）
            feature_points = self.detect_feature_points(skeleton)
            print(f"去重后特征点：{len(feature_points)}个（端点+交点）")
            
            # 3. 追踪线段（含合并）
            segments = self.trace_wire_segments(skeleton, feature_points)
            print(f"合并后导线线段：{len(segments)}条")
            
            # 4. 线段宽度分析和噪声过滤
            filtered_segments = segments
            width_stats_list = []
            
            if original_image is not None and len(segments) > 0:
                print("开始线段宽度分析...")
                
                # 计算所有线段的宽度统计信息
                width_stats_list = []
                for i, (start, end, path) in enumerate(segments):
                    width_stats = self.calculate_segment_width_statistics(path, original_image)
                    width_stats_list.append(width_stats)
                    print(f"  线段 {i+1}: 平均宽度 {width_stats['avg_width']:.2f}, "
                          f"最常见宽度 {width_stats['most_common_width']:.2f}")
                
                # 基于宽度统计进行过滤
                if len(segments) > 1:
                    # 可视化宽度分布
                    self.filter_segments_by_width_statistics(
                        segments, width_stats_list, threshold_ratio=2.0, show_plot=True)
                    
                    # 实际过滤
                    filtered_segments, filtered_stats = self.filter_segments_by_width_statistics(
                        segments, width_stats_list, threshold_ratio=2.0, show_plot=False)
                    print(f"宽度过滤后线段数: {len(filtered_segments)}条 (原始: {len(segments)}条)")
                    width_stats_list = filtered_stats  # 更新统计信息
                else:
                    print("线段数量不足，跳过宽度过滤")
            else:
                print("未提供原始图像或线段数量为0，跳过宽度分析")
            
            # 5. 构建连接图（使用过滤后的线段）
            connection_graph = self.build_connection_graph(filtered_segments)
            print(f"连接图节点数：{connection_graph.number_of_nodes()}，边数：{connection_graph.number_of_edges()}")
            
            # 6. 分析线段连接关系（使用过滤后的线段）
            connections = []
            if components is not None and texts is not None:
                connections = self.find_connected_components_or_texts(filtered_segments, components, texts)
            
            # 7. 保存可视化结果（显示过滤后的线段）
            skeleton_path, result_path = self.visualize_results(
                img, skeleton, feature_points, filtered_segments, output_dir, image_filename
            )
            
            # 8. 生成详细连接关系可视化
            # 如果提供了原始图像，则使用原始图像进行详细可视化；否则使用骨架图像
            visualization_image = original_image if original_image is not None else img
            if connections and visualization_image is not None:
                self.generate_detailed_visualizations(connections, visualization_image, output_dir, image_filename)
                # 生成连接关系的markdown摘要
                self.generate_connection_markdown(connections, output_dir, image_filename)
            
            return {
                'feature_points': feature_points,
                'segments': filtered_segments,  # 返回过滤后的线段
                'original_segments': segments,  # 保留原始线段信息
                'width_stats': width_stats_list,  # 宽度统计信息
                'connection_graph': connection_graph,
                'connections': connections,  # 新增连接信息
                'skeleton_path': skeleton_path,
                'result_path': result_path
            }
            
        except Exception as e:
            print(f"线连接检测出错：{e}")
            return None