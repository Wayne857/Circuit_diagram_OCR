import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt
from pathlib import Path


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
        """合并端点距离小于阈值的线段（避免一条线拆成多条）"""
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
                    if (self.distance(end_i, start_j) < threshold or 
                        self.distance(end_i, end_j) < threshold or
                        self.distance(start_i, start_j) < threshold or
                        self.distance(start_i, end_j) < threshold):
                    
                        # 合并两条线段的路径
                        if self.distance(end_i, start_j) < threshold:
                            new_path = path_i + path_j
                            new_start = start_i
                            new_end = end_j
                        elif self.distance(end_i, end_j) < threshold:
                            new_path = path_i + path_j[::-1]
                            new_start = start_i
                            new_end = start_j
                        elif self.distance(start_i, start_j) < threshold:
                            new_path = path_j + path_i
                            new_start = start_j
                            new_end = end_i
                        else: # distance(start_i, end_j) < threshold
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
        """追踪导线线段（优化BFS，允许微小偏移）"""
        h, w = skeleton.shape
        visited = set()
        segments = []
        point_coords = {p[1] for p in feature_points}

        def bfs(start_point):
            queue = [start_point]
            path = [start_point]
            visited.add(start_point)
            while queue:
                x, y = queue.pop(0)
                # 检查8邻域（增加容错：允许1像素偏移）
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < w and 0 <= ny_ < h:
                            if skeleton[ny_, nx_] == 255 and (nx_, ny_) not in visited:
                                visited.add((nx_, ny_))
                                path.append((nx_, ny_))
                                queue.append((nx_, ny_))
                                # 遇到特征点则终止，但允许微小距离容错
                                if (nx_, ny_) in point_coords:
                                    return path
            return path

        # 追踪所有特征点
        for point_type, point in feature_points:
            if point not in visited:
                path = bfs(point)
                if len(path) >= 2:
                    segments.append((path[0], path[-1], path))
        
        # 合并距离极近的线段（核心优化：避免一条线拆成多条）
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

    def detect_line_connections(self, image_path, output_dir, image_filename):
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
            
            # 4. 构建连接图
            connection_graph = self.build_connection_graph(segments)
            print(f"连接图节点数：{connection_graph.number_of_nodes()}，边数：{connection_graph.number_of_edges()}")
            
            # 5. 保存可视化结果
            skeleton_path, result_path = self.visualize_results(
                img, skeleton, feature_points, segments, output_dir, image_filename
            )
            
            return {
                'feature_points': feature_points,
                'segments': segments,
                'connection_graph': connection_graph,
                'skeleton_path': skeleton_path,
                'result_path': result_path
            }
            
        except Exception as e:
            print(f"线连接检测出错：{e}")
            return None