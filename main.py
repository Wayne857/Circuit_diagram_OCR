#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主程序文件
功能：加载YOLO模型，执行目标检测，并将选定类别的检测框区域变为纯白色
"""

import sys
import argparse
from pathlib import Path
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from utils.image_processor import ImageProcessor
from utils.config import Config

def get_device():
    """自动检测可用设备"""
    if torch.cuda.is_available():
        return 0  # 使用第一个GPU
    else:
        return 'cpu'  # 使用CPU

def process_single_image(model, image_processor, image_path, selected_classes, process_type, output_dir):
    """处理单张图像"""
    print(f"\n处理图像: {image_path}")
    
    try:
        # 确保输出目录存在
        output_dir = Path(output_dir)
        detection_output_dir = output_dir / "detection_results"
        processed_output_dir = output_dir / "processed_images"
        detection_output_dir.mkdir(parents=True, exist_ok=True)
        processed_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 执行目标检测
        detection_results = model.predict(
            task="predict",
            source=str(image_path),
            conf=0.25,
            iou=0.45,
            save=True,  # 保存检测结果
            save_txt=False,
            save_conf=False,
            show=False,
            device=get_device()
        )
        
        # 生成输出路径
        image_name = Path(image_path).stem
        image_ext = Path(image_path).suffix
        detection_result_path = detection_output_dir / f"{image_name}_result{image_ext}"
        processed_image_path = processed_output_dir / f"{image_name}_whitened{image_ext}"
        
        # 处理检测结果，将选定类别的检测框区域进行处理
        image_processor.process_detection_results(
            str(image_path), 
            detection_results, 
            selected_classes, 
            str(processed_image_path),
            process_type
        )
        
        print(f"  检测结果已保存到: {detection_result_path}")
        print(f"  处理结果已保存到: {processed_image_path}")
        
        return True
        
    except Exception as e:
        print(f"  处理图像时出错: {e}")
        return False

def process_folder_images(model, image_processor, input_folder, selected_classes, process_type, output_dir):
    """处理文件夹中的所有图像"""
    # 初始化配置
    config = Config()
    
    # 获取输入图像列表
    input_images = config.get_input_images()
    
    if not input_images:
        print(f"警告: 在 {input_folder} 目录中未找到图像文件")
        return False
    
    print(f"找到 {len(input_images)} 个待处理图像")
    
    success_count = 0
    # 处理每个图像
    for i, image_path in enumerate(input_images, 1):
        print(f"\n处理图像 ({i}/{len(input_images)}): {image_path}")
        
        try:
            # 执行目标检测
            detection_results = model.predict(
                task="predict",
                source=image_path,
                conf=0.25,
                iou=0.45,
                save=True,  # 保存检测结果
                save_txt=False,
                save_conf=False,
                show=False,
                device=get_device()
            )
            
            # 获取输出路径
            detection_result_path, processed_image_path = config.get_output_paths(image_path)
            
            # 处理检测结果，将选定类别的检测框区域进行处理
            image_processor.process_detection_results(
                image_path, 
                detection_results, 
                selected_classes, 
                processed_image_path,
                process_type
            )
            
            print(f"  检测结果已保存到: {detection_result_path}")
            print(f"  处理结果已保存到: {processed_image_path}")
            success_count += 1
            
        except Exception as e:
            print(f"  处理图像时出错: {e}")
            continue
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(input_images)} 张图像")
    print(f"结果已保存到 {config.output_dir} 目录中")
    return True

def process_folder_images_with_segmentation(detection_model, segmentation_model, image_processor, input_folder, detection_classes_to_remove, segmentation_conf, output_dir):
    """处理文件夹中的所有图像，先检测并移除指定类别，然后进行分割"""
    # 初始化配置
    config = Config()
    
    # 获取输入图像列表
    input_images = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for file in input_folder.iterdir():
        if file.suffix.lower() in image_extensions:
            input_images.append(str(file))
    
    if not input_images:
        print(f"警告: 在 {input_folder} 目录中未找到图像文件")
        return False
    
    print(f"找到 {len(input_images)} 个待处理图像")
    
    success_count = 0
    # 处理每个图像
    for i, image_path in enumerate(input_images, 1):
        print(f"\n处理图像 ({i}/{len(input_images)}): {image_path}")
        
        try:
            # 对每张图像执行检测、移除指定类别并分割
            image_processor.process_detection_and_segmentation(
                image_path, 
                detection_model, 
                segmentation_model, 
                detection_classes_to_remove, 
                output_dir,
                segmentation_conf
            )
            success_count += 1
            
        except Exception as e:
            print(f"  处理图像时出错: {e}")
            continue
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(input_images)} 张图像")
    return True

def main():
    """主函数"""
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='YOLO目标检测、去除特定类别并进行分割')
    parser.add_argument('--image', '-i', type=str, help='单张图像路径')
    parser.add_argument('--folder', '-f', type=str, help='图像文件夹路径')
    parser.add_argument('--output', '-o', type=str, default='predict_res', help='输出目录路径')
    parser.add_argument('--detection-classes', '-dc', type=int, nargs='+', default=[1], help='要从检测结果中移除的类别ID列表 (默认: [1])')
    parser.add_argument('--segmentation-conf', '-sc', type=float, default=0.5, help='分割置信度阈值 (默认: 0.5)')
    parser.add_argument('--mode', '-m', type=str, default='detect_and_segment', choices=['detect_and_segment', 'detect_only'], 
                       help='运行模式: detect_and_segment(检测并分割) 或 detect_only(仅检测) (默认: detect_and_segment)')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.image and not args.folder:
        print("错误: 必须指定 --image 或 --folder 参数")
        print("使用 --help 查看详细用法")
        return
    
    # 初始化配置
    config = Config()
    
    # 检查模型文件是否存在
    if not config.detection_model_path.exists():
        print(f"错误: 检测模型文件不存在: {config.detection_model_path}")
        print("请先训练检测模型或检查模型路径")
        return
    
    if not config.segmentation_model_path.exists():
        print(f"错误: 分割模型文件不存在: {config.segmentation_model_path}")
        print("请先训练分割模型或检查模型路径")
        return
    
    # 初始化图像处理器
    image_processor = ImageProcessor()
    
    # 加载训练好的模型
    print(f"正在加载检测模型: {config.detection_model_path}")
    detection_model = YOLO(str(config.detection_model_path))
    print("检测模型加载成功!")
    
    print(f"正在加载分割模型: {config.segmentation_model_path}")
    segmentation_model = YOLO(str(config.segmentation_model_path))
    print("分割模型加载成功!")
    
    # 获取设备信息
    device = get_device()
    print(f"使用设备: {device}")
    
    # 获取处理参数
    detection_classes_to_remove = args.detection_classes
    segmentation_conf = args.segmentation_conf
    output_dir = args.output
    mode = args.mode
    
    print(f"将从检测结果中移除以下类别的检测框: {detection_classes_to_remove}")
    print(f"分割置信度阈值: {segmentation_conf}")
    print(f"输出目录: {output_dir}")
    print(f"运行模式: {mode}")
    
    # 处理单张图像
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"错误: 图像文件不存在: {image_path}")
            return
        
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            print(f"错误: 不支持的图像格式: {image_path.suffix}")
            return
            
        if mode == 'detect_and_segment':
            # 检测并分割模式
            image_processor.process_detection_and_segmentation(
                str(image_path), 
                detection_model, 
                segmentation_model, 
                detection_classes_to_remove, 
                output_dir,
                segmentation_conf
            )
        else:
            # 仅检测模式（保持原有功能）
            process_single_image(detection_model, image_processor, image_path, detection_classes_to_remove, 'whiten', output_dir)
    
    # 处理文件夹中的图像
    elif args.folder:
        input_folder = Path(args.folder)
        if not input_folder.exists():
            print(f"错误: 文件夹不存在: {input_folder}")
            return
            
        if mode == 'detect_and_segment':
            # 检测并分割模式
            process_folder_images_with_segmentation(detection_model, segmentation_model, image_processor, input_folder, detection_classes_to_remove, segmentation_conf, output_dir)
        else:
            # 仅检测模式（保持原有功能）
            process_folder_images(detection_model, image_processor, input_folder, detection_classes_to_remove, 'whiten', output_dir)

if __name__ == "__main__":
    main()