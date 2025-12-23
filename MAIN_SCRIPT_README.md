# Main脚本增强版使用说明

## 概述

本项目中的main.py脚本已更新，支持先进行目标检测并移除特定类别，然后对处理后的图像进行分割的功能。

## 功能特点

- 先使用检测模型识别目标并移除指定类别（如text类）
- 对处理后的图像使用分割模型进行分割
- 按要求组织输出结果

## 输出目录结构

运行脚本后，会创建如下目录结构：

```
output_directory/
├── original/                    # 原始图像
│   └── image_name_original.jpg
├── processed_after_detection/   # 检测后移除指定类别的图像
│   └── image_name_after_detection.jpg
├── segmented/                   # 分割结果图像
│   └── image_name_segmented.jpg
└── classes/                     # 按类别分割的图像
    ├── arrow/
    ├── capacitor/
    ├── chip/
    ├── ground/
    ├── line/
    ├── line_connector/
    ├── motor/
    ├── resistor/
    ├── zener_diode/
    ├── mov/
    ├── fuse/
    └── inductor/
```

## 使用方法

### 1. 命令行参数

```bash
python main.py --image "图像路径" --detection-classes "要移除的类别ID" --mode "detect_and_segment" --output "输出目录"
```

参数说明：
- `--image`: 输入图像路径（单张图像模式）
- `--folder`: 输入图像文件夹路径（批量处理模式）
- `--output`: 输出目录路径
- `--detection-classes`: 要从检测结果中移除的类别ID列表（默认为[1]，通常是text类）
- `--segmentation-conf`: 分割置信度阈值（默认为0.5）
- `--mode`: 运行模式，`detect_and_segment`（检测并分割）或 `detect_only`（仅检测）

### 2. 运行示例

处理单张图像：
```bash
python main.py --image "imagessegment/seg/images/val/AC-DC Voltage Conversion_LS03-13B12R3 V3.png" --output "runs/segment/predict_detailed_main" --detection-classes 1 --mode detect_and_segment
```

处理整个文件夹：
```bash
python main.py --folder "imagessegment/seg/images/val" --output "runs/segment/predict_detailed_folder" --detection-classes 1 --mode detect_and_segment
```

## 支持的类别

### 检测模型类别（用于移除）
- 类别1：通常是text类（根据您的模型配置）

### 分割模型类别
- arrow (0)
- capacitor (1)
- chip (2)
- ground (3)
- line (4)
- line_connector (5)
- motor (6)
- resistor (7)
- zener_diode (8)
- mov (9)
- fuse (10)
- inductor (11)

## 注意事项

1. 确保两个模型文件都存在：
   - 检测模型：`runs/detect/train/weights/best.pt`
   - 分割模型：`runs/segment/train24/weights/best.pt`

2. 检测到的指定类别会被替换为白色区域

3. 输出目录会自动创建，如果已存在同名文件会被覆盖

4. 分割结果会按类别保存到对应子目录中

## 模式说明

- `detect_and_segment`：先检测并移除指定类别，然后对处理后的图像进行分割（推荐）
- `detect_only`：仅执行原有的检测和处理功能

该脚本实现了您要求的完整流程：原始图像 → 检测 → 移除text类 → 分割 → 按类别保存结果。