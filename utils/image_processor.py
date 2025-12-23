import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
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
    
    def whiten_areas(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        将掩码区域变为纯白色
        
        Args:
            image (np.ndarray): 原始图像
            mask (np.ndarray): 掩码图像
            
        Returns:
            np.ndarray: 处理后的图像
        """
        # 创建白色图像
        white_image = np.ones_like(image) * 255
        
        # 使用掩码合并原图和白色图像
        result = np.where(mask[..., None] == 255, white_image, image)
        
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
        处理检测结果，将选定类别的检测框区域变为纯白色
        
        Args:
            image_path (str): 原始图像路径
            detection_results: YOLO检测结果对象
            selected_classes (List[int]): 选中的类别ID列表
            output_path (str): 输出图像路径
            process_type (str): 处理类型，"whiten" 或 "blur"
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
            result_image = self.whiten_areas(image, mask)
        elif process_type == "blur":
            result_image = self.blur_areas(image, mask)
        else:
            result_image = self.whiten_areas(image, mask)  # 默认使用白化
        
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
        
        # 类别名称定义
        class_names = {
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
        for class_id, class_name in class_names.items():
            class_dir = classes_dir / class_name
            class_dir.mkdir(exist_ok=True)
            class_dirs[class_id] = class_dir
        
        # 保存原始图像
        original_output_path = original_dir / f"{image_filename}_original.jpg"
        self.save_image(original_image, str(original_output_path))
        
        # 执行检测
        detection_results = detection_model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            save=False,
            show=False
        )
        
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
        # 将需要移除的区域设为白色
        processed_image[mask_to_remove == 255] = [255, 255, 255]
        
        # 保存处理后的图像
        processed_output_path = processed_dir / f"{image_filename}_after_detection.jpg"
        self.save_image(processed_image, str(processed_output_path))
        
        # 对处理后的图像进行分割
        segmentation_results = segmentation_model(processed_image, conf=segmentation_conf)
        
        # 保存分割的整体结果
        if segmentation_results and len(segmentation_results) > 0:
            seg_result = segmentation_results[0]
            
            # 创建整体分割图
            if seg_result.masks is not None:
                # 获取原图用于叠加
                annotated_img = seg_result.plot()  # 包含分割掩码和边界框的图像
                segmented_output_path = segmented_dir / f"{image_filename}_segmented.jpg"
                self.save_image(annotated_img, str(segmented_output_path))
                
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
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    
                    # 保存该类别的分割结果
                    class_output_path = class_dirs[class_id] / f"{image_filename}_{class_name}_instance_{i+1}_conf_{confidence:.2f}.jpg"
                    self.save_image(class_result, str(class_output_path))
                    
                    print(f"  保存类别 {class_name} 实例 {i+1}，置信度 {confidence:.2f}")
            else:
                # 如果没有掩码，只保存处理后的图像
                segmented_output_path = segmented_dir / f"{image_filename}_segmented.jpg"
                self.save_image(processed_image, str(segmented_output_path))
        
        print(f"  分割结果已保存到: {output_path}")