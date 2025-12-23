import os
import numpy as np
import json
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from os import getcwd
from collections import defaultdict

def get_file(json_path, test_size=0.1):
    """获取JSON文件列表并划分训练集和测试集"""
    files = glob(json_path + "*.json")
    files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
    if test_size > 0:
        trainval_files, test_files = train_test_split(files, test_size=test_size, random_state=55)
    else:
        trainval_files = files
        test_files = []
    return trainval_files, test_files

def convert(size, box):
    """将边界框坐标转换为YOLO格式"""
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def get_all_classes(json_path):
    """从所有JSON文件中提取所有可能的类别"""
    classes = set()
    files = glob(json_path + "*.json")
    
    for json_file in files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for shape in data["shapes"]:
                classes.add(shape["label"])
    
    return sorted(list(classes))

def count_labels_by_class(json_path, files):
    """统计每个类别的标签数量"""
    class_count = defaultdict(int)
    
    for json_file_ in files:
        json_filename = os.path.join(json_path, json_file_ + ".json")
        if not os.path.exists(json_filename):
            continue
            
        with open(json_filename, "r", encoding="utf-8") as f:
            try:
                json_file = json.load(f)
                for multi in json_file["shapes"]:
                    label = multi["label"]
                    class_count[label] += 1
            except Exception as e:
                print(f"处理文件 {json_filename} 时出错: {e}")
                
    return dict(class_count)

def json_to_txt(json_path, img_path, files, output_path, txt_name, classes):
    """
    将JSON标签文件转换为YOLO格式的txt文件
    参数:
    - json_path: JSON文件路径
    - img_path: 图像文件路径
    - files: 文件名列表
    - output_path: 输出路径
    - txt_name: 输出txt文件名
    - classes: 类别列表
    """
    # 创建输出目录
    labels_path = os.path.join(output_path, "labels")
    images_path = os.path.join(output_path, "images")
    
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    
    # 创建图像列表文件
    list_file = open(os.path.join(output_path, '%s.txt' % (txt_name)), 'w')
    
    # 导入tqdm显示进度
    try:
        from tqdm import tqdm
        files_iter = tqdm(files, desc=f"处理{txt_name}文件")
    except ImportError:
        print("未安装tqdm库，将不显示进度条")
        files_iter = files
    
    for json_file_ in files_iter:
        # JSON文件路径
        json_filename = os.path.join(json_path, json_file_ + ".json")
        # 查找图像文件（支持jpg和png格式）
        img_extensions = [".jpg", ".png"]
        img_filename = None
        for ext in img_extensions:
            possible_img_path = os.path.join(img_path, json_file_ + ext)
            if os.path.exists(possible_img_path):
                img_filename = possible_img_path
                break
        
        if img_filename is None:
            print(f"警告: 找不到 {json_file_} 对应的图像文件")
            continue
            
        # 将图像路径写入列表文件
        list_file.write('%s\n' % (img_filename))
        
        # 读取图像获取尺寸
        img = cv2.imread(img_filename)
        if img is None:
            print(f"警告: 无法读取图像 {img_filename}")
            continue
        height, width, channels = img.shape
        
        # 创建对应的txt标签文件
        label_filename = os.path.join(labels_path, json_file_ + ".txt")
        out_file = open(label_filename, 'w')
        
        # 读取JSON文件
        with open(json_filename, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        
        # 处理每个标注
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            # 获取矩形框的坐标
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]

            # 检查边界框是否有效
            if xmax <= xmin or ymax <= ymin:
                continue
                
            # 检查标签是否在类别列表中
            if label not in classes:
                print(f"警告: 标签 '{label}' 不在类别列表中")
                continue
                
            cls_id = classes.index(label)
            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            bb = convert((width, height), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

        out_file.close()
        
        # 复制图像到images目录
        dst_img_path = os.path.join(images_path, os.path.basename(img_filename))
        if not os.path.exists(dst_img_path):
            import shutil
            shutil.copyfile(img_filename, dst_img_path)

    list_file.close()

if __name__ == '__main__':
    # 当前工作目录
    wd = getcwd()
    
    # 定义输入路径
    json_path = "imagesdetect/json/"
    img_path = "imagesdetect/img/"
    
    # 自动获取所有类别
    classes_name = get_all_classes(json_path)
    print(f"检测到的类别: {classes_name}")
    
    # 获取文件列表并划分数据集
    train_files, val_files = get_file(json_path, test_size=0.2)  # 20%作为验证集
    
    # 统计每个类别的标签数量
    print("统计训练集各类别标签数量...")
    train_class_count = count_labels_by_class(json_path, train_files)
    print("统计验证集各类别标签数量...")
    val_class_count = count_labels_by_class(json_path, val_files)
    
    # 显示统计结果
    print("\n训练集类别统计:")
    for cls in classes_name:
        count = train_class_count.get(cls, 0)
        print(f"  {cls}: {count}")
        
    print("\n验证集类别统计:")
    for cls in classes_name:
        count = val_class_count.get(cls, 0)
        print(f"  {cls}: {count}")
    
    # 创建detect输出目录
    detect_output_path = "imagesdetect/detect"
    if not os.path.exists(detect_output_path):
        os.makedirs(detect_output_path)
    
    # 创建训练集和验证集目录
    train_output_path = os.path.join(detect_output_path, "train")
    val_output_path = os.path.join(detect_output_path, "val")
    
    # 生成训练集
    print("\n正在生成训练集...")
    json_to_txt(json_path, img_path, train_files, train_output_path, "train", classes_name)
    
    # 生成验证集
    print("\n正在生成验证集...")
    json_to_txt(json_path, img_path, val_files, val_output_path, "val", classes_name)
    
    print("\n数据集转换完成！")
    print(f"训练集数量: {len(train_files)}")
    print(f"验证集数量: {len(val_files)}")
    print(f"类别数量: {len(classes_name)}")
    # yolo detect train data=C:\Users\11\Desktop\pj\image_extract\ultralytics\ultralytics\cfg\datasets\coco.yaml model=C:\Users\11\Desktop\pj\image_extract\ultralytics\ultralytics\cfg\detect_model\yolov8x.pt epochs=200 imgsz=720 batch=8 workers=0 device=0
