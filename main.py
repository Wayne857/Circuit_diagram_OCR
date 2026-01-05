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
from utils.text_processor import TextProcessor
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
            save=False,  # 不自动保存检测结果
            show=False,
            device=get_device()
        )
        
        # 提取并保存检测框内的图像区域
        original_image = image_processor.load_image(str(image_path))
        image_filename = Path(image_path).stem
        
        # 创建检测结果目录
        detection_results_dir = output_dir / "detection_results"
        detection_results_dir.mkdir(parents=True, exist_ok=True)
        
        # 2类别模型的类别名称定义
        class_names_2class = {
            0: 'connector',
            1: 'text'
        }
        
        # 12类别模型的类别名称定义
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
                    image_processor.save_image(cropped_img, str(cropped_img_path))
                    
                    print(f"  保存检测框图像: {cropped_img_path} (类别: {class_name}, 置信度: {conf:.2f})")
        
        # 生成输出路径
        image_name = Path(image_path).stem
        image_ext = Path(image_path).suffix
        processed_image_path = processed_output_dir / f"{image_name}_whitened{image_ext}"
        
        # 处理检测结果，将选定类别的检测框区域进行处理
        image_processor.process_detection_results(
            str(image_path), 
            detection_results, 
            selected_classes, 
            str(processed_image_path),
            process_type
        )
        
        # 检测结果会自动保存到 {output_dir}/detection_results/ 目录中
        detection_result_path = output_dir / "detection_results" / f"{image_name}{image_ext}"
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
            text_results = text_processor.recognize_texts_concurrent(text_cropped_images_data, max_workers=5)
            
            # 保存文本识别结果到txt文件
            text_result_file = Path(output_dir) / "text_results.txt"
            mode = 'w'  # 使用写入模式，因为这是单个图像处理
            with open(text_result_file, mode, encoding='utf-8') as f:
                f.write("文本识别结果:\n")  # 添加文件头
                for result in text_results:
                    if result['success']:
                        # 解析文本识别结果
                        data = result['data']
                        if 'parsing_res_list' in data and data['parsing_res_list']:
                            for item in data['parsing_res_list']:
                                text_content = item.get('text', '') if isinstance(item, dict) else str(item)
                                f.write(f"图像名称: {result['image_name']}, 坐标: {result['bbox_coords']}, 识别文本: {text_content}\n")
                        else:
                            f.write(f"图像名称: {result['image_name']}, 坐标: {result['bbox_coords']}, 识别文本: {data}\n")
                    else:
                        f.write(f"图像名称: {result['image_name']}, 坐标: {result['bbox_coords']}, 识别失败: {result.get('error', 'Unknown error')}\n")
            print(f"  文本识别结果已保存到: {text_result_file}")
        
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
                save=False,  # 不自动保存检测结果
                show=False,
                device=get_device()
            )
            
            # 提取并保存检测框内的图像区域
            original_image = image_processor.load_image(image_path)
            image_filename = Path(image_path).stem
            
            # 创建检测结果目录
            detection_results_dir = Path(output_dir) / "detection_results"
            detection_results_dir.mkdir(parents=True, exist_ok=True)
            
            # 2类别模型的类别名称定义
            class_names_2class = {
                0: 'text',
                1: 'connector'
            }
            
            # 12类别模型的类别名称定义
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
                        image_processor.save_image(cropped_img, str(cropped_img_path))
                        
                        print(f"  保存检测框图像: {cropped_img_path} (类别: {class_name}, 置信度: {conf:.2f})")
            
            # 处理检测结果，将选定类别的检测框区域进行处理
            image_processor.process_detection_results(
                image_path, 
                detection_results, 
                selected_classes, 
                processed_image_path,
                process_type
            )
            
            # 检测结果会自动保存到 {output_dir}/detection_results/ 目录中
            detection_result_path = Path(output_dir) / "detection_results" / Path(image_path).name
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
                text_results = text_processor.recognize_texts_concurrent(text_cropped_images_data, max_workers=5)
                
                # 保存文本识别结果到txt文件
                text_result_file = Path(output_dir) / "text_results.txt"
                mode = 'a' if i > 0 else 'w'  # 第一个图像使用写入模式，后续使用追加模式
                with open(text_result_file, mode, encoding='utf-8') as f:
                    if mode == 'w':
                        f.write("文本识别结果:\n")  # 添加文件头
                    for result in text_results:
                        if result['success']:
                            # 解析文本识别结果
                            data = result['data']
                            if 'parsing_res_list' in data and data['parsing_res_list']:
                                for item in data['parsing_res_list']:
                                    text_content = item.get('text', '') if isinstance(item, dict) else str(item)
                                    f.write(f"图像名称: {result['image_name']}, 坐标: {result['bbox_coords']}, 识别文本: {text_content}\n")
                            else:
                                f.write(f"图像名称: {result['image_name']}, 坐标: {result['bbox_coords']}, 识别文本: {data}\n")
                        else:
                            f.write(f"图像名称: {result['image_name']}, 坐标: {result['bbox_coords']}, 识别失败: {result.get('error', 'Unknown error')}\n")
                print(f"  文本识别结果已保存到: {text_result_file}")
            
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