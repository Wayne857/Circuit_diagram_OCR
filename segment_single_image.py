import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


def segment_single_image(image_path, model_path, output_dir, conf=0.5):
    """
    使用指定模型对单张图像进行分割，并保存详细结果
    包括整体分割图、原图以及按类别分割的各个部分
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    original_dir = output_path / "original"
    segmented_dir = output_path / "segmented"
    classes_dir = output_path / "classes"
    
    original_dir.mkdir(exist_ok=True)
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
    
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image_filename = Path(image_path).stem
    
    # 进行预测
    results = model(image_path, conf=conf)
    
    # 保存原图
    original_output_path = original_dir / f"{image_filename}_original.jpg"
    cv2.imwrite(str(original_output_path), image)
    
    # 复制原图到输出目录根
    main_output_path = output_path / f"{image_filename}_segmented.jpg"
    
    # 如果有分割结果
    if results and len(results) > 0:
        result = results[0]
        
        # 创建整体分割图
        if result.masks is not None:
            # 获取原图用于叠加
            annotated_img = result.plot()  # 包含分割掩码和边界框的图像
            cv2.imwrite(str(main_output_path), annotated_img)
            
            # 保存整体分割图
            segmented_output_path = segmented_dir / f"{image_filename}_segmented.jpg"
            cv2.imwrite(str(segmented_output_path), annotated_img)
            
            # 处理每个检测到的分割实例
            for i, (mask, cls, conf) in enumerate(zip(result.masks.xy, result.boxes.cls, result.boxes.conf)):
                class_id = int(cls)
                confidence = float(conf)
                
                # 创建该实例的掩码
                h, w = image.shape[:2]
                mask_img = np.zeros((h, w), dtype=np.uint8)
                
                # 填充掩码区域
                if len(mask) > 0:
                    mask_points = np.array([mask], dtype=np.int32)
                    cv2.fillPoly(mask_img, mask_points, 255)
                
                # 提取该类别的分割部分
                class_mask = (mask_img > 0).astype(np.uint8) * 255
                class_mask_3ch = cv2.cvtColor(class_mask, cv2.COLOR_GRAY2BGR)
                
                # 创建该实例的掩码图像
                class_result = np.zeros_like(image)
                class_result = np.where(class_mask_3ch == 255, image, 0)
                
                # 获取类别名称
                class_name = class_names.get(class_id, f"class_{class_id}")
                
                # 保存该类别的分割结果
                class_output_path = class_dirs[class_id] / f"{image_filename}_{class_name}_instance_{i+1}_conf_{confidence:.2f}.jpg"
                cv2.imwrite(str(class_output_path), class_result)
                
                print(f"Saved class {class_name} instance {i+1} with confidence {confidence:.2f}")
        else:
            # 如果没有掩码，只保存原图
            cv2.imwrite(str(main_output_path), image)
            segmented_output_path = segmented_dir / f"{image_filename}_segmented.jpg"
            cv2.imwrite(str(segmented_output_path), image)
    
    print(f"Segmentation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Segment Single Image with YOLOv8')
    parser.add_argument('--image', '-i', type=str, required=True, help='Input image path')
    parser.add_argument('--model', '-m', type=str, default='runs/segment/train24/weights/best.pt', 
                       help='Model path (default: runs/segment/train24/weights/best.pt)')
    parser.add_argument('--output', '-o', type=str, default='runs/segment/predict_detailed', 
                       help='Output directory path')
    parser.add_argument('--conf', '-c', type=float, default=0.5, 
                       help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    segment_single_image(args.image, args.model, args.output, args.conf)


if __name__ == "__main__":
    main()