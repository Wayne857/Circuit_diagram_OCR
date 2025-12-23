# 分割结果详细处理脚本使用说明

## 概述

本项目包含用于处理YOLOv8分割模型结果的脚本，可以将train24模型的分割结果按类别详细组织并保存。

## 文件说明

- `segment_single_image.py` - 处理单张图像的分割脚本
- `segment_folder.py` - 处理整个文件夹图像的分割脚本
- `run_segmentation.bat` - Windows批处理运行脚本
- `SEGMENTATION_README.md` - 本说明文档

## 输出目录结构

运行脚本后，会创建如下目录结构：

```
output_directory/
├── original/                    # 原始图像
│   └── image_name_original.jpg
├── segmented/                   # 整体分割结果图像
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

### 1. 处理单张图像

```bash
python segment_single_image.py --image "path/to/image.jpg" --model "runs/segment/train24/weights/best.pt" --output "output/directory" --conf 0.5
```

参数说明：
- `--image`: 输入图像路径
- `--model`: 模型路径（默认为train24模型）
- `--output`: 输出目录路径
- `--conf`: 置信度阈值（默认为0.5）

### 2. 处理整个文件夹

```bash
python segment_folder.py --folder "path/to/image/folder" --model "runs/segment/train24/weights/best.pt" --output "output/directory" --conf 0.5
```

参数说明：
- `--folder`: 输入图像文件夹路径
- `--model`: 模型路径（默认为train24模型）
- `--output`: 输出目录路径
- `--conf`: 置信度阈值（默认为0.5）

### 3. 使用批处理脚本

直接运行 `run_segmentation.bat` 文件，它会自动处理示例图像。

## 支持的类别

脚本支持以下12个类别：
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

1. 确保模型权重文件存在：`runs/segment/train24/weights/best.pt`
2. 图像文件路径中包含空格时，请使用引号包围路径
3. 输出目录会自动创建，如果已存在同名文件会被覆盖
4. 每个分割实例会根据其置信度保存，并在文件名中包含置信度信息

## 示例

运行以下命令处理验证集中的所有图像：

```bash
python segment_folder.py --folder "imagessegment/seg/images/val" --model "runs/segment/train24/weights/best.pt" --output "runs/segment/predict_detailed" --conf 0.5
```

这将为验证集中的每张图像创建详细的分割结果，按上述目录结构组织。