"""
将YOLO格式的标签转换为像素级分割标签图像
"""
import os
import cv2
import numpy as np
from pathlib import Path

# 定义类别映射
class_names = ['resistor', 'motor', 'ground', 'line', 'arrow', 'line_connector', 'chip', 'capacitor', 'zener_diode', 'mov', 'fuse', 'inductor']
num_classes = len(class_names)

def yolo_to_segmentation_mask(image_path, txt_path, output_path, img_shape):
    """
    将YOLO格式的标签转换为分割标签图像
    
    Args:
        image_path: 原始图像路径
        txt_path: YOLO标签文件路径
        output_path: 输出分割标签图像路径
        img_shape: 图像形状 (height, width)
    """
    height, width = img_shape
    
    # 创建全黑的分割标签图像 (值为0)
    seg_mask = np.zeros((height, width), dtype=np.uint8)
    
    if not os.path.exists(txt_path):
        # 如果没有标签文件，则返回全零图像
        cv2.imwrite(output_path, seg_mask)
        return
    
    # 读取YOLO格式的标签
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析YOLO格式: class_id x_center y_center width height
        parts = line.split()
        if len(parts) != 5:
            continue
            
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width_ratio = float(parts[3])
        height_ratio = float(parts[4])
        
        # 转换为像素坐标
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        width_px = int(width_ratio * width)
        height_px = int(height_ratio * height)
        
        # 计算边界框坐标
        x1 = max(0, int(x_center_px - width_px / 2))
        y1 = max(0, int(y_center_px - height_px / 2))
        x2 = min(width, int(x_center_px + width_px / 2))
        y2 = min(height, int(y_center_px + height_px / 2))
        
        # 在分割标签图像中填充对应类别的值
        seg_mask[y1:y2, x1:x2] = class_id + 1  # 类别ID从1开始，0表示背景

def convert_dataset(image_dir, label_dir, output_dir):
    """
    批量转换数据集
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))
    
    for img_path in image_files:
        img_name = img_path.stem  # 获取文件名（不含扩展名）
        
        # 对应的标签文件
        txt_path = os.path.join(label_dir, img_name + '.txt')
        
        # 输出的分割标签图像路径
        output_path = os.path.join(output_dir, img_name + '.png')
        
        # 读取原始图像以获取尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
            
        height, width = img.shape[:2]
        
        # 转换标签
        yolo_to_segmentation_mask(str(img_path), txt_path, output_path, (height, width))
        print(f"已转换: {img_name}")

if __name__ == "__main__":
    # 定义路径
    base_path = "C:/Users/11/Desktop/pj/image_extract/imagessegment/seg"
    
    # 转换训练集
    print("正在转换训练集...")
    convert_dataset(
        image_dir=os.path.join(base_path, "images", "train"),
        label_dir=os.path.join(base_path, "labels", "train"),
        output_dir=os.path.join(base_path, "seg_labels", "train")
    )
    
    # 转换验证集
    print("正在转换验证集...")
    convert_dataset(
        image_dir=os.path.join(base_path, "images", "val"),
        label_dir=os.path.join(base_path, "labels", "val"),
        output_dir=os.path.join(base_path, "seg_labels", "val")
    )
    
    print("转换完成！")
    
    # 更新配置文件指向正确的标签目录
    print("请将配置文件中的 ann_dir 从 'labels/train' 和 'labels/val' 改为 'seg_labels/train' 和 'seg_labels/val'")