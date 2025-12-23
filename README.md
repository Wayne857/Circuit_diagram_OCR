# 电路元件检测与分割处理系统

本项目是一个完整的电路元件图像处理系统，支持目标检测、语义分割、图像处理等多种功能。系统能够检测电路图中的元件，对特定类别进行处理，并使用深度学习模型进行精确分割。

## 项目概述

本项目包含以下核心功能：
- **目标检测**：使用YOLO模型检测电路图中的各类元件
- **图像处理**：对检测结果进行处理（如白化、模糊等）
- **语义分割**：使用训练好的分割模型对图像进行像素级分割
- **工作流程**：支持检测→移除特定类别→分割的完整流程

## 项目结构

```
image_extract/
├── main.py                    # 主程序：检测、移除类别、分割一体化处理
├── utils/                     # 工具模块
│   ├── config.py             # 配置管理类
│   └── image_processor.py    # 图像处理类
├── detect_json2txt.py         # JSON标注转YOLO格式脚本
├── segment_json2txt.py        # JSON标注转分割格式脚本
├── predict_detect.py          # 目标检测预测脚本
├── predict_seg.py             # 语义分割预测脚本
├── predict_seg_detailed.py    # 详细分割结果处理脚本
├── segment_single_image.py    # 单图像分割脚本
├── segment_folder.py          # 批量图像分割脚本
├── run_segmentation.bat       # Windows分割运行脚本
├── detect_menu.bat            # 检测功能菜单脚本
├── run_detect.bat             # 简单检测运行脚本
├── imagessegment/             # 分割数据集目录
│   ├── json/                 # JSON标注文件
│   ├── txt/                  # YOLO格式标注
│   └── seg/                  # 分割数据集
├── imagesdetect/              # 检测数据集目录
│   ├── json/                 # JSON标注文件
│   └── detect/               # 检测数据集
├── runs/                      # 训练结果目录
│   ├── detect/               # 检测模型训练结果
│   └── segment/              # 分割模型训练结果
├── ultralytics/               # YOLO框架目录
├── dataset_SEG.py             # 数据集处理脚本
├── convert_yolo_to_seg.py     # YOLO转分割格式转换器
├── yolo_to_seg_converter.py   # YOLO到语义分割转换器
└── README.md                 # 项目说明文档
```

## 核心功能

### 1. 目标检测
- 支持12种电路元件检测
  - 0: arrow (箭头)
  - 1: capacitor (电容)
  - 2: chip (芯片)
  - 3: ground (地线)
  - 4: line (连线)
  - 5: line_connector (连接器)
  - 6: motor (电机)
  - 7: resistor (电阻)
  - 8: zener_diode (齐纳二极管)
  - 9: mov (压敏电阻)
  - 10: fuse (保险丝)
  - 11: inductor (电感)
- 可指定特定类别进行处理
- 支持白化和模糊两种处理方式

### 2. 语义分割
- 使用训练好的分割模型进行像素级分割
- 支持对处理后的图像进行分割
- 按类别保存分割结果

### 3. 完整处理流程
- 检测图像中的元件
- 移除指定类别（如text类）
- 对处理后的图像进行分割
- 按目录结构保存结果

## 使用方法

### 主程序使用

```bash
# 单图像处理
python main.py --image "图像路径" --detection-classes 1 --mode detect_and_segment --output "输出目录"

# 批量处理
python main.py --folder "图像文件夹" --detection-classes 1 --mode detect_and_segment --output "输出目录"

# 查看所有选项
python main.py --help
```

### 分割功能使用

```bash
# 单图像分割
python segment_single_image.py --image "图像路径" --model "模型路径" --output "输出目录"

# 批量分割
python segment_folder.py --folder "图像文件夹" --model "模型路径" --output "输出目录"
```

## 输出目录结构

处理结果按以下结构组织：

```
输出目录/
├── original/                    # 原始图像
├── processed_after_detection/   # 检测后处理的图像
├── segmented/                   # 分割结果图像
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

## 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- 其他常用Python库

## 项目特点

1. **模块化设计**：使用utils模块组织工具类
2. **灵活配置**：支持命令行参数配置
3. **完整流程**：从检测到分割的一体化处理
4. **结果组织**：清晰的输出目录结构
5. **批处理支持**：支持单张和批量处理
6. **Windows友好**：提供批处理脚本支持

## 技术栈

- **深度学习框架**：Ultralytics YOLO
- **图像处理**：OpenCV
- **编程语言**：Python
- **架构设计**：面向对象设计模式
