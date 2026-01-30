import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
from .text_processor import TextProcessor
from .fasttext_classifier import FastTextComponentClassifier
from .text_component_matcher import TextComponentMatcher
import os
from pathlib import Path

class ImageProcessor:
    """图像处理类，用于处理YOLO检测结果"""
    
    def __init__(self):
        """初始化图像处理器"""
        pass
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            np.ndarray: 图像数组
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return image
    
    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """
        保存图像
        
        Args:
            image (np.ndarray): 图像数组
            output_path (str): 输出路径
        """
        cv2.imwrite(output_path, image)
    
    def create_mask_from_boxes(self, image_shape: Tuple[int, int], boxes: List[List[float]]) -> np.ndarray:
        """
        根据检测框创建掩码
        
        Args:
            image_shape (Tuple[int, int]): 图像形状 (height, width)
            boxes (List[List[float]]): 检测框列表 [[x1, y1, x2, y2], ...]
            
        Returns:
            np.ndarray: 掩码图像
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, image_shape[1]))
            x2 = max(0, min(x2, image_shape[1]))
            y1 = max(0, min(y1, image_shape[0]))
            y2 = max(0, min(y2, image_shape[0]))
            
            # 在掩码上绘制矩形区域
            mask[y1:y2, x1:x2] = 255
            
        return mask
    
    def colorize_areas(self, image: np.ndarray, mask: np.ndarray, color_type: str = "white") -> np.ndarray:
        """
        将掩码区域变为指定颜色（白色或背景色）
        
        Args:
            image (np.ndarray): 原始图像
            mask (np.ndarray): 掩码图像
            color_type (str): 颜色类型，"white" 或 "background"
            
        Returns:
            np.ndarray: 处理后的图像
        """
        if color_type == "white":
            # 创建白色图像
            colored_image = np.ones_like(image) * 255
        elif color_type == "background":
            # 使用背景色
            from .background_detector import get_background_color_advanced
            bg_color = get_background_color_advanced(image)
            if len(image.shape) == 3:
                colored_image = np.full_like(image, bg_color)
            else:
                colored_image = np.full_like(image, bg_color, dtype=image.dtype)
        else:
            # 默认使用白色
            colored_image = np.ones_like(image) * 255
        
        # 使用掩码合并原图和着色图像
        result = np.where(mask[..., None] == 255, colored_image, image)
        
        return result
    
    def blur_areas(self, image: np.ndarray, mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """
        对掩码区域进行模糊处理（可选的替代方案）
        
        Args:
            image (np.ndarray): 原始图像
            mask (np.ndarray): 掩码图像
            kernel_size (int): 模糊核大小
            
        Returns:
            np.ndarray: 处理后的图像
        """
        # 对整个图像进行模糊处理
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # 使用掩码合并原图和模糊图像
        result = np.where(mask[..., None] == 255, blurred_image, image)
        
        return result
    
    def process_detection_results(self, image_path: str, detection_results, selected_classes: List[int], 
                               output_path: str, process_type: str = "whiten") -> None:
        """
        处理检测结果，将选定类别的检测框区域变为纯白色或背景色
        
        Args:
            image_path (str): 原始图像路径
            detection_results: YOLO检测结果对象
            selected_classes (List[int]): 选中的类别ID列表
            output_path (str): 输出图像路径
            process_type (str): 处理类型，"whiten", "background_color", 或 "blur"
        """
        # 加载原始图像
        image = self.load_image(image_path)
        
        # 提取选中类别的检测框
        boxes = []
        for result in detection_results:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
            classes = result.boxes.cls.cpu().numpy()      # 获取类别ID
            confidences = result.boxes.conf.cpu().numpy() # 获取置信度
            
            # 筛选出选中类别的边界框
            for box, cls_id, conf in zip(boxes_xyxy, classes, confidences):
                if int(cls_id) in selected_classes:
                    boxes.append(box.tolist())
                    print(f"  处理类别 {int(cls_id)} (置信度: {conf:.2f}) 的检测框: {box}")
        
        if not boxes:
            print("  未找到选中类别的检测框")
            # 保存原始图像
            self.save_image(image, output_path)
            return
        
        print(f"  共处理 {len(boxes)} 个检测框")
        
        # 创建掩码
        mask = self.create_mask_from_boxes(image.shape, boxes)
        
        # 根据处理类型选择处理方法
        if process_type == "whiten":
            result_image = self.colorize_areas(image, mask, "white")
        elif process_type == "background_color":
            result_image = self.colorize_areas(image, mask, "background")
        elif process_type == "blur":
            result_image = self.blur_areas(image, mask)
        else:
            result_image = self.colorize_areas(image, mask, "white")  # 默认使用白化
        
        # 保存结果图像
        self.save_image(result_image, output_path)
    
    def process_detection_and_segmentation(self, image_path: str, detection_model, segmentation_model, 
                                          detection_classes_to_remove: List[int], output_dir: str,
                                          segmentation_conf: float = 0.5) -> None:
        """
        先进行目标检测并去除指定类别，然后对剩余图像进行分割
        
        Args:
            image_path (str): 原始图像路径
            detection_model: 检测模型
            segmentation_model: 分割模型
            detection_classes_to_remove (List[int]): 需要移除的检测类别ID列表
            output_dir (str): 输出目录
            segmentation_conf (float): 分割置信度阈值
        """
        # 加载原始图像
        original_image = self.load_image(image_path)
        image_filename = Path(image_path).stem
        
        # 创建输出目录结构
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        original_dir = output_path / "original"
        processed_dir = output_path / "processed_after_detection"
        segmented_dir = output_path / "segmented"
        classes_dir = output_path / "classes"
        
        original_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)
        segmented_dir.mkdir(exist_ok=True)
        classes_dir.mkdir(exist_ok=True)
        
        # 2类别模型的类别名称定义（用于detection_results目录）
        class_names_2class = {
            0: 'connector',
            1: 'text'
        }
        
        # 12类别模型的类别名称定义（用于classes目录）
        class_names_12class = {
            0: 'arrow',
            1: 'capacitor', 
            2: 'chip',
            3: 'ground',
            4: 'line',
            5: 'line_connector',
            6: 'motor',
            7: 'resistor',
            8: 'zener_diode',
            9: 'mov',
            10: 'fuse',
            11: 'inductor'
        }
        
        # 为每个类别创建子目录
        class_dirs = {}
        for class_id, class_name in class_names_12class.items():
            class_dir = classes_dir / class_name
            class_dir.mkdir(exist_ok=True)
            class_dirs[class_id] = class_dir
        
        # 保存原始图像
        original_output_path = original_dir / f"{image_filename}_original.jpg"
        self.save_image(original_image, str(original_output_path))
        
        # 创建检测结果目录
        detection_results_dir = output_path / "detection_results"
        detection_results_dir.mkdir(exist_ok=True)
        
        # 执行检测
        detection_results = detection_model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            save=False,  # 不自动保存检测结果
            show=False
        )
        
        # 提取并保存检测框内的图像区域
        for i, result in enumerate(detection_results):
            boxes = result.boxes
            if boxes is not None:
                for j, (box, conf) in enumerate(zip(boxes.xyxy, boxes.conf)):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(boxes.cls[j]) if boxes.cls is not None else 0
                    
                    # 根据class_names_2class确定类别名称
                    class_name = class_names_2class.get(class_id, f"class_{class_id}")
                    
                    # 如果是text类别，扩大检测框
                    if class_name == 'text':
                        # 计算扩大后的坐标
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        new_width = int(width * 1.2)
                        new_height = int(height * 1.2)
                        
                        x1_new = int(center_x - new_width / 2)
                        x2_new = int(center_x + new_width / 2)
                        y1_new = int(center_y - new_height / 2)
                        y2_new = int(center_y + new_height / 2)
                        
                        # 确保坐标在图像范围内
                        x1_new = max(0, x1_new)
                        y1_new = max(0, y1_new)
                        x2_new = min(original_image.shape[1], x2_new)
                        y2_new = min(original_image.shape[0], y2_new)
                        
                        # 提取扩大后的检测框内的图像区域
                        cropped_img = original_image[y1_new:y2_new, x1_new:x2_new]
                    else:
                        # 提取原始检测框内的图像区域
                        cropped_img = original_image[y1:y2, x1:x2]
                    
                    # 创建类别子目录并保存裁剪的图像
                    class_detection_dir = detection_results_dir / class_name
                    class_detection_dir.mkdir(exist_ok=True)
                    
                    cropped_img_path = class_detection_dir / f"{image_filename}_detected_{class_name}_{i+1}_{j+1}_conf_{conf:.2f}.jpg"
                    self.save_image(cropped_img, str(cropped_img_path))
                    
                    print(f"  保存检测框图像: {cropped_img_path} (类别: {class_name}, 置信度: {conf:.2f})")
        
        # 创建掩码以移除指定类别
        mask_to_remove = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        for result in detection_results:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标
            classes = result.boxes.cls.cpu().numpy()      # 获取类别ID
            confidences = result.boxes.conf.cpu().numpy() # 获取置信度
            
            # 筛选出需要移除的类别
            for box, cls_id, conf in zip(boxes_xyxy, classes, confidences):
                if int(cls_id) in detection_classes_to_remove:
                    x1, y1, x2, y2 = map(int, box)
                    # 确保坐标在图像范围内
                    x1 = max(0, min(x1, original_image.shape[1]))
                    x2 = max(0, min(x2, original_image.shape[1]))
                    y1 = max(0, min(y1, original_image.shape[0]))
                    y2 = max(0, min(y2, original_image.shape[0]))
                    
                    # 在掩码上绘制矩形区域以标记需要移除的部分
                    mask_to_remove[y1:y2, x1:x2] = 255
                    print(f"  移除类别 {int(cls_id)} (置信度: {conf:.2f}) 的检测框: {box}")
        
        # 创建处理后的图像（移除指定类别）
        processed_image = original_image.copy()
        
        # 计算背景色
        from .background_detector import get_background_color_advanced
        bg_color = get_background_color_advanced(original_image)
        
        # 将需要移除的区域设为背景色（而不是白色）
        if len(processed_image.shape) == 3:
            bg_mask = np.full_like(processed_image, bg_color)
        else:
            bg_mask = np.full_like(processed_image, bg_color, dtype=processed_image.dtype)
        processed_image[mask_to_remove == 255] = bg_mask[mask_to_remove == 255]
        
        # 保存处理后的图像
        processed_output_path = processed_dir / f"{image_filename}_after_detection.jpg"
        self.save_image(processed_image, str(processed_output_path))
        
        # 对处理后的图像进行分割
        segmentation_results = segmentation_model(processed_image, conf=segmentation_conf, iou=0.45)
        
        # 收集分割结果
        segmentation_info = []
        
        # 创建segmented_out目录，用于保存去掉分割内容的图像
        segmented_out_dir = output_path / "segmented_out"
        segmented_out_dir.mkdir(exist_ok=True)
        
        # 保存分割的整体结果
        if segmentation_results and len(segmentation_results) > 0:
            seg_result = segmentation_results[0]
            
            # 创建整体分割图
            if seg_result.masks is not None:
                # 获取原图用于叠加
                annotated_img = seg_result.plot()  # 包含分割掩码和边界框的图像
                segmented_output_path = segmented_dir / f"{image_filename}_segmented.jpg"
                self.save_image(annotated_img, str(segmented_output_path))
                
                # 创建一个副本用于移除分割内容
                image_without_segments = processed_image.copy()
                
                # 处理每个检测到的分割实例
                for i, (mask, cls, conf) in enumerate(zip(seg_result.masks.xy, seg_result.boxes.cls, seg_result.boxes.conf)):
                    class_id = int(cls)
                    confidence = float(conf)
                    
                    # 创建该实例的掩码
                    h, w = processed_image.shape[:2]
                    mask_img = np.zeros((h, w), dtype=np.uint8)
                    
                    # 填充掩码区域
                    if len(mask) > 0:
                        mask_points = np.array([mask], dtype=np.int32)
                        cv2.fillPoly(mask_img, mask_points, 255)
                    
                    # 提取该类别的分割部分
                    class_mask = (mask_img > 0).astype(np.uint8) * 255
                    class_mask_3ch = cv2.cvtColor(class_mask, cv2.COLOR_GRAY2BGR)
                    
                    # 创建该实例的掩码图像
                    class_result = np.zeros_like(processed_image)
                    class_result = np.where(class_mask_3ch == 255, processed_image, 0)
                    
                    # 获取类别名称
                    class_name = class_names_12class.get(class_id, f"class_{class_id}")
                    
                    # 根据类别设置不同的颜色
                    # line类别保持不变，chip类别设置为红色，其他类别设置为背景色
                    if class_name == 'chip':
                        # chip类别设置为红色
                        red_color = [0, 0, 255]  # BGR格式的红色
                        if len(processed_image.shape) == 3:
                            color_mask = np.full_like(processed_image, red_color)
                        else:
                            color_mask = np.full_like(processed_image, red_color[2], dtype=processed_image.dtype)  # 灰度图取蓝色通道值
                        image_without_segments = np.where(class_mask_3ch == 255, color_mask, image_without_segments)
                    elif class_name != 'line':  # 其他非line类别设置为背景色
                        # 导入背景色检测工具
                        from .background_detector import get_background_color_advanced
                        bg_color = get_background_color_advanced(processed_image)
                        if len(processed_image.shape) == 3:
                            bg_mask = np.full_like(processed_image, bg_color)
                        else:
                            bg_mask = np.full_like(processed_image, bg_color, dtype=processed_image.dtype)
                        image_without_segments = np.where(class_mask_3ch == 255, bg_mask, image_without_segments)
                    
                    # 获取检测框坐标
                    if seg_result.boxes is not None and len(seg_result.boxes) > i:
                        box = seg_result.boxes.xyxy[i]
                        bbox_coords = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    else:
                        bbox_coords = None
                    
                    # 保存分割坐标信息
                    segmentation_info.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox_coords': bbox_coords,
                        'mask_coords': mask.astype(int).tolist() if len(mask) > 0 else [],
                        'confidence': confidence
                    })
                    
                    # 保存该类别的分割结果
                    class_output_path = class_dirs[class_id] / f"{image_filename}_{class_name}_instance_{i+1}_conf_{confidence:.2f}.jpg"
                    self.save_image(class_result, str(class_output_path))
                    
                    print(f"  保存类别 {class_name} 实例 {i+1}，置信度 {confidence:.2f}")
                
                # 保存去掉分割内容的图像
                segmented_out_path = segmented_out_dir / f"{image_filename}_without_segments.jpg"
                self.save_image(image_without_segments, str(segmented_out_path))
                print(f"  保存去掉分割内容的图像: {segmented_out_path}")
                
                # 使用霍夫直线检测，将检测到的直线变成红色
                # 转换为灰度图
                gray = cv2.cvtColor(image_without_segments, cv2.COLOR_BGR2GRAY)
                
                # 应用边缘检测
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                
                # 使用霍夫变换检测直线
                lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
                
                # 创建图像副本用于绘制直线
                hough_lines_img = image_without_segments.copy()
                
                if lines is not None:
                    for rho, theta in lines[:, 0]:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        
                        # 绘制红色直线
                        cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 保存带有红色直线的图像
                hough_output_path = segmented_out_dir / f"{image_filename}_with_hough_lines.jpg"
                self.save_image(hough_lines_img, str(hough_output_path))
                print(f"  保存带有霍夫直线的图像: {hough_output_path}")
            else:
                # 如果没有掩码，只保存处理后的图像
                segmented_output_path = segmented_dir / f"{image_filename}_segmented.jpg"
                self.save_image(processed_image, str(segmented_output_path))
                
                # 保存原始处理后的图像到segmented_out目录（因为没有分割内容可以移除）
                segmented_out_path = segmented_out_dir / f"{image_filename}_without_segments.jpg"
                self.save_image(processed_image, str(segmented_out_path))
                print(f"  保存图像到segmented_out目录: {segmented_out_path}")
        
        # 初始化文本处理器并识别text类别的文本
        text_processor = TextProcessor()
        
        # 收集所有text类别的裁剪图像数据
        text_cropped_images_data = []
        for i, result in enumerate(detection_results):
            boxes = result.boxes
            if boxes is not None:
                for j, (box, conf) in enumerate(zip(boxes.xyxy, boxes.conf)):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(boxes.cls[j]) if boxes.cls is not None else 0
                    
                    # 根据class_names_2class确定类别名称
                    class_name = class_names_2class.get(class_id, f"class_{class_id}")
                    
                    # 如果是text类别，准备进行文本识别
                    if class_name == 'text':
                        # 如果是text类别，扩大检测框
                        if class_name == 'text':
                            # 计算扩大后的坐标
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            new_width = int(width * 1.2)
                            new_height = int(height * 1.2)
                            
                            x1_new = int(center_x - new_width / 2)
                            x2_new = int(center_x + new_width / 2)
                            y1_new = int(center_y - new_height / 2)
                            y2_new = int(center_y + new_height / 2)
                            
                            # 确保坐标在图像范围内
                            x1_new = max(0, x1_new)
                            y1_new = max(0, y1_new)
                            x2_new = min(original_image.shape[1], x2_new)
                            y2_new = min(original_image.shape[0], y2_new)
                            
                            # 提取扩大后的检测框内的图像区域
                            cropped_img = original_image[y1_new:y2_new, x1_new:x2_new]
                        else:
                            # 提取原始检测框内的图像区域
                            cropped_img = original_image[y1:y2, x1:x2]
                        
                        # 添加到待处理列表
                        text_cropped_images_data.append({
                            'image': cropped_img,
                            'name': f"{image_filename}_text_{i+1}_{j+1}",
                            'bbox': (x1, y1, x2, y2),  # 保存原始坐标
                            'confidence': float(conf)
                        })
        
        # 如果有text类别的图像，进行并发文本识别
        if text_cropped_images_data:
            print(f"  开始识别 {len(text_cropped_images_data)} 个text类别的图像")
            text_results = text_processor.recognize_texts_concurrent(text_cropped_images_data, max_workers=5, output_dir=output_path)
            
            # 保存文本识别结果到txt文件
            text_result_file = output_path / "text_results.txt"
            
            # 初始化FastText分类器
            try:
                # 使用绝对路径来确保模型文件可以被找到
                import os
                model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'component_fasttext.bin')
                classifier = FastTextComponentClassifier(model_path)
                
                # 初始化分类结果列表
                classification_results = []
                
                # 收集所有成功识别的文本内容，用于分类
                texts_to_classify = []
                texts_to_process = []  # 保存需要处理的result对象
                
                for result in text_results:
                    if result['success']:
                        extracted_content = result.get('extracted_content', str(result.get('data', '')))
                        # 检查是否符合正则匹配规则，如果是则直接分类
                        import re
                        text_lower = extracted_content.lower()
                        
                        # 检查特定符号进行直接分类
                        if 'Ω' in extracted_content:  # 包含Ω符号直接判定为电阻
                            classification_results.append({'predicted_label': '电阻', 'confidence': 1.0})
                        elif 'μf' in text_lower:  # 包含μF直接判定为电容
                            classification_results.append({'predicted_label': '电容', 'confidence': 1.0})
                        elif 'mh' in text_lower or 'μh' in text_lower:  # 包含mH或μH直接判定为电感
                            classification_results.append({'predicted_label': '电感', 'confidence': 1.0})
                        elif re.match(r'^\d+k$', text_lower):  # 以k结尾且前面都是数字的判定为电阻
                            classification_results.append({'predicted_label': '电阻', 'confidence': 1.0})
                        else:
                            # 不符合正则匹配规则的文本才进行FastText分类
                            texts_to_classify.append(extracted_content)
                            texts_to_process.append(result)  # 保存对应的result对象
                
                # 批量进行FastText分类预测（仅对不符合正则规则的文本）
                if texts_to_classify:
                    fasttext_classification_results = classifier.predict_batch(texts_to_classify)
                    
                    # 将FastText分类结果插入到对应位置
                    regular_idx = 0  # 指向下一个正则匹配结果
                    fasttext_idx = 0  # 指向下一个FastText结果
                    
                    # 重建完整的classification_results列表，按原始顺序
                    full_classification_results = []
                    for result in text_results:
                        if result['success']:
                            extracted_content = result.get('extracted_content', str(result.get('data', '')))
                            text_lower = extracted_content.lower()
                            
                            # 检查是否是正则匹配的文本
                            if ('Ω' in extracted_content or 'μf' in text_lower or 
                                'mh' in text_lower or 'μh' in text_lower or 
                                re.match(r'^\d+k$', text_lower)):
                                # 正则匹配的文本，使用预先设定的结果
                                # 由于我们已经将这些结果放在了classification_results中，需要按顺序取出
                                # 但更好的方式是重新生成
                                if 'Ω' in extracted_content:
                                    full_classification_results.append({'predicted_label': '电阻', 'confidence': 1.0})
                                elif 'μf' in text_lower:
                                    full_classification_results.append({'predicted_label': '电容', 'confidence': 1.0})
                                elif 'mh' in text_lower or 'μh' in text_lower:
                                    full_classification_results.append({'predicted_label': '电感', 'confidence': 1.0})
                                elif re.match(r'^\d+k$', text_lower):
                                    full_classification_results.append({'predicted_label': '电阻', 'confidence': 1.0})
                            else:
                                # FastText分类的文本
                                if fasttext_idx < len(fasttext_classification_results):
                                    full_classification_results.append(fasttext_classification_results[fasttext_idx])
                                    fasttext_idx += 1
                                else:
                                    full_classification_results.append(None)
                    
                    classification_results = full_classification_results
                else:
                    # 如果所有文本都通过正则匹配分类了，我们需要确保classification_results是完整且有序的
                    full_classification_results = []
                    for result in text_results:
                        if result['success']:
                            extracted_content = result.get('extracted_content', str(result.get('data', '')))
                            text_lower = extracted_content.lower()
                            
                            if 'Ω' in extracted_content:
                                full_classification_results.append({'predicted_label': '电阻', 'confidence': 1.0})
                            elif 'μf' in text_lower:
                                full_classification_results.append({'predicted_label': '电容', 'confidence': 1.0})
                            elif 'mh' in text_lower or 'μh' in text_lower:
                                full_classification_results.append({'predicted_label': '电感', 'confidence': 1.0})
                            elif re.match(r'^\d+k$', text_lower):
                                full_classification_results.append({'predicted_label': '电阻', 'confidence': 1.0})
                            else:
                                full_classification_results.append(None)
                    
                    classification_results = full_classification_results
            except Exception as e:
                print(f"  FastText分类器初始化或预测失败: {e}")
                classification_results = [None] * len([r for r in text_results if r['success']])
            
            # 写入结果文件
            with open(text_result_file, 'w', encoding='utf-8') as f:
                f.write("文本识别与分类结果:\n")  # 更新文件头
                
                # 跟踪分类结果的索引
                classification_idx = 0
                
                for result in text_results:
                    if result['success']:
                        # 使用text_processor中提取的内容
                        extracted_content = result.get('extracted_content', str(result.get('data', '')))
                        
                        # 获取对应的分类结果
                        classification_result = None
                        if classification_idx < len(classification_results):
                            classification_result = classification_results[classification_idx]
                            classification_idx += 1
                        
                        # 写入文本结果和坐标
                        f.write(f"文本结果: {extracted_content}, 文本坐标: {result['bbox_coords']}\n")
                        
                        # 如果分类成功，写入分类结果
                        if classification_result:
                            f.write(f"分类结果: {classification_result['predicted_label']}, 置信度: {classification_result['confidence']}\n")
                        else:
                            f.write(f"分类结果: 未分类\n")
                    else:
                        f.write(f"图像名称: {result['image_name']}, 坐标: {result['bbox_coords']}, 识别失败: {result.get('error', 'Unknown error')}\n")
                
                # 添加分割结果
                if segmentation_info:
                    f.write("\n分割结果:\n")
                    for seg_info in segmentation_info:
                        f.write(f"类别: {seg_info['class_name']}, 检测框坐标: {seg_info['bbox_coords']}, 分割坐标: {seg_info['mask_coords'][:10]}... (置信度: {seg_info['confidence']:.2f})\n")  # 只显示前10个坐标点以避免输出过长
                            
                # 收集文本识别结果用于匹配
                text_data = []
                for result in text_results:
                    if result['success']:
                        extracted_content = result.get('extracted_content', str(result.get('data', '')))
                        # 根据文本内容推断类别
                        import re
                        text_lower = extracted_content.lower()
                        
                        # 检查特定符号进行直接分类
                        if 'Ω' in extracted_content:  # 包含Ω符号直接判定为电阻
                            text_category = "电阻"
                        elif 'μf' in text_lower:  # 包含μF直接判定为电容
                            text_category = "电容"
                        elif 'mh' in text_lower or 'μh' in text_lower:  # 包含mH或μH直接判定为电感
                            text_category = "电感"
                        elif re.match(r'^\d+k$', text_lower):  # 以k结尾且前面都是数字的判定为电阻
                            text_category = "电阻"
                        elif any(keyword in text_lower for keyword in ['r', 'ω', 'ohm', 'kΩ', 'mΩ']):
                            text_category = "电阻"
                        elif any(keyword in text_lower for keyword in ['c', 'μf', 'nf', 'pf', 'farad']):
                            text_category = "电容"
                        elif any(keyword in text_lower for keyword in ['gnd', 'ground', 'earth']):
                            text_category = "接地"
                        elif any(keyword in text_lower for keyword in ['u', 'ic', 'chip', 'tps', 'stm', 'arduino']):
                            text_category = "芯片"
                        elif 'nc' in text_lower:
                            text_category = "芯片"  # NC (No Connection) 通常与芯片引脚相关
                        elif 'ct' in text_lower:
                            text_category = "电容"  # CT (Capacitor Tag) 通常与电容相关
                        else:
                            text_category = "芯片"  # 默认类别
                        text_data.append({
                            'text': extracted_content,
                            'coord': result['bbox_coords'],
                            'category': text_category,
                            'conf': 0.8  # 假设文本识别置信度
                        })
                            
                # 收集元件检测结果用于匹配
                component_data = []
                for seg_info in segmentation_info:
                    if seg_info['bbox_coords']:
                        component_data.append({
                            'category': seg_info['class_name'],
                            'bbox': seg_info['bbox_coords'],
                            'conf': seg_info['confidence'],
                            'segmentation': seg_info['mask_coords']
                        })
                            
                # 如果有文本和元件数据，执行匹配
                if text_data and component_data:
                    matcher = TextComponentMatcher()
                    matches, _, _ = matcher.optimal_text_component_matching(text_data, component_data)
                                
                    # 将匹配结果写入文件
                    match_result_str = matcher.format_match_results(matches)
                    f.write(match_result_str)
                                
                    # 保存可视化结果到run目录
                    try:
                        visualization_path = matcher.save_visualization(matches, text_data, str(output_path))
                        print(f"  匹配可视化结果已保存到: {visualization_path}")
                    except Exception as e:
                        print(f"  保存可视化结果时出错: {e}")
                        # 如果出错，尝试直接保存到output_path目录
                        try:
                            visualization_path = str(Path(output_path) / "matching_result.png")
                            matcher.visualize_matching_result(matches, text_data, 800, 600, visualization_path)
                            print(f"  匹配可视化结果已保存到: {visualization_path}")
                        except Exception as e2:
                            print(f"  备用保存方式也失败: {e2}")
            print(f"  文本识别结果已保存到: {text_result_file}")
            
            # 线连接检测：对去掉分割内容的图像进行线连接分析
            try:
                from .line_connection_detector import LineConnectionDetector
                detector = LineConnectionDetector()
                
                # 创建connect_process目录
                connect_process_dir = output_path / "connect_process"
                
                # 获取去掉分割内容的图像路径
                without_segments_path = segmented_out_dir / f"{image_filename}_without_segments.jpg"
                
                if without_segments_path.exists():
                    # 准备元件和文本数据用于线连接分析
                    components = []
                    texts = []
                    
                    # 从分割结果中提取元件数据
                    for seg_info in segmentation_info:
                        if seg_info['bbox_coords']:
                            components.append({
                                'category': seg_info['class_name'],
                                'bbox': seg_info['bbox_coords'],
                                'conf': seg_info['confidence'],
                                'segmentation': seg_info['mask_coords']
                            })
                    
                    # 从文本结果中提取文本数据
                    for result in text_results:
                        if result['success']:
                            extracted_content = result.get('extracted_content', str(result.get('data', '')))
                            texts.append({
                                'text': extracted_content,
                                'coord': result['bbox_coords'],
                                'conf': 0.8  # 假设文本识别置信度
                            })
                    
                    # 读取原始图像用于可视化（彩色图像，保留元件和文本的颜色信息）
                    original_without_segments = cv2.imread(str(without_segments_path))
                    
                    # 执行线连接检测
                    result = detector.detect_line_connections(
                        str(without_segments_path), 
                        str(connect_process_dir), 
                        image_filename,
                        components=components,
                        texts=texts,
                        original_image=original_without_segments  # 传递原始图像用于可视化
                    )
                    
                    if result:
                        print(f"  线连接检测完成，结果已保存到: {connect_process_dir}")
                        print(f"  检测到 {len(result['segments'])} 条导线段")
                        print(f"  检测到 {len(result['feature_points'])} 个特征点")
                        print(f"  分析了 {len(result['connections'])} 个连接关系")
                    else:
                        print("  线连接检测失败")
                else:
                    print(f"  警告: 未找到去掉分割内容的图像: {without_segments_path}")
                    
            except ImportError:
                print("  警告: 无法导入LineConnectionDetector，跳过线连接检测")
            except Exception as e:
                print(f"  线连接检测出错: {e}")
        
        print(f"  检测框内的图像已保存到: {output_path}/detection_results/")
        print(f"  分割结果已保存到: {output_path}")