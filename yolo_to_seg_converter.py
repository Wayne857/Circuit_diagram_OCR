import os
import cv2
import numpy as np
from pathlib import Path
import yaml

def convert_yolo_to_seg(yolo_label_path, image_path, output_path, img_shape):
    """
    将YOLO格式的多边形标签转换为语义分割PNG格式
    """
    h, w = img_shape[:2]
    
    # 创建空白标签图像
    seg_mask = np.zeros((h, w), dtype=np.uint8)
    
    if os.path.exists(yolo_label_path):
        with open(yolo_label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            values = line.strip().split()
            if len(values) < 6:  # 至少需要 class_id x1 y1 x2 y2 x3 y3
                continue
                
            class_id = int(values[0])
            coords = [float(x) for x in values[1:]]
            
            # 检查坐标数量是否为偶数（x,y对）
            if len(coords) % 2 != 0:
                print(f"警告: {yolo_label_path} 中的坐标数量不是偶数，跳过此行: {line}")
                continue
            
            # 将归一化坐标转换为像素坐标
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
            
            if len(points) >= 3:  # 至少需要3个点形成多边形
                points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                
                # 填充多边形区域
                cv2.fillPoly(seg_mask, [points], color=(class_id))
    
    # 保存为PNG格式
    cv2.imwrite(output_path, seg_mask)

def create_seg_dataset(yolo_dataset_path, output_path):
    """
    将整个YOLO数据集转换为语义分割格式
    """
    yolo_path = Path(yolo_dataset_path)
    output_path = Path(output_path)
    
    # 创建输出目录结构
    (output_path / 'img_dir' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'img_dir' / 'val').mkdir(parents=True, exist_ok=True)
    (output_path / 'ann_dir' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'ann_dir' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 读取数据集配置
    config_path = yolo_path / 'segment.yaml'
    if not config_path.exists():
        print(f"错误: 找不到配置文件 {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 转换训练集
    print("正在转换训练集...")
    train_img_dir = yolo_path / config['train']
    train_label_dir = yolo_path / 'labels' / 'train'
    
    if not train_img_dir.exists():
        print(f"错误: 找不到训练图像目录 {train_img_dir}")
        return
    
    if not train_label_dir.exists():
        print(f"错误: 找不到训练标签目录 {train_label_dir}")
        return
    
    img_count = 0
    for img_file in train_img_dir.glob('*.png'):
        # 使用numpy读取图像，以更好地处理中文路径
        img_absolute_path = str(img_file.resolve())
        img = cv2.imdecode(np.fromfile(img_absolute_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is not None:
            # 复制图像文件 - 保持原始文件名
            output_img_path = output_path / 'img_dir' / 'train' / img_file.name
            # 用imencode来写入图像，这样可以避免中文路径问题
            success, encoded_img = cv2.imencode('.png', img)
            if success:
                with open(output_img_path, 'wb') as f:
                    f.write(encoded_img)
            
            # 转换标签文件
            label_file = train_label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                seg_output_path = output_path / 'ann_dir' / 'train' / f"{img_file.stem}.png"
                convert_yolo_to_seg(str(label_file), str(img_file), str(seg_output_path), img.shape)
            else:
                print(f"警告: 找不到标签文件 {label_file}")
            
            img_count += 1
            if img_count % 50 == 0:
                print(f"已处理 {img_count} 个训练图像")
        else:
            print(f"无法读取图像: {img_file}")
    
    print(f"训练集转换完成，共处理 {img_count} 个图像")
    
    # 转换验证集
    print("正在转换验证集...")
    val_path = config.get('val', 'images/val')
    val_img_dir = yolo_path / val_path
    val_label_dir = yolo_path / 'labels' / 'val'
    
    if val_img_dir.exists() and val_label_dir.exists():
        img_count = 0
        for img_file in val_img_dir.glob('*.png'):
            # 使用numpy读取图像，以更好地处理中文路径
            img_absolute_path = str(img_file.resolve())
            img = cv2.imdecode(np.fromfile(img_absolute_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if img is not None:
                # 复制图像文件 - 保持原始文件名
                output_img_path = output_path / 'img_dir' / 'val' / img_file.name
                # 用imencode来写入图像，这样可以避免中文路径问题
                success, encoded_img = cv2.imencode('.png', img)
                if success:
                    with open(output_img_path, 'wb') as f:
                        f.write(encoded_img)
                
                # 转换标签文件
                label_file = val_label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    seg_output_path = output_path / 'ann_dir' / 'val' / f"{img_file.stem}.png"
                    convert_yolo_to_seg(str(label_file), str(img_file), str(seg_output_path), img.shape)
                else:
                    print(f"警告: 找不到验证集标签文件 {label_file}")
                
                img_count += 1
                if img_count % 50 == 0:
                    print(f"已处理 {img_count} 个验证图像")
            else:
                print(f"无法读取验证集图像: {img_file}")
        
        print(f"验证集转换完成，共处理 {img_count} 个图像")
    else:
        print(f"验证集目录不存在，跳过验证集转换: {val_img_dir}, {val_label_dir}")
    
    # 保存类别信息
    classes = list(config['names'].values())
    print(f"转换完成！类别数：{len(classes)}，类别名称：{classes}")
    
    # 创建类别名称文件
    with open(output_path / 'classes.txt', 'w', encoding='utf-8') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

if __name__ == "__main__":
    # 指定YOLO数据集路径和输出路径
    yolo_dataset_path = r"C:\Users\11\Desktop\pj\image_extract\imagessegment\seg"
    output_path = r"C:\Users\11\Desktop\pj\image_extract\segman_dataset_fixed"
    create_seg_dataset(yolo_dataset_path, output_path)
