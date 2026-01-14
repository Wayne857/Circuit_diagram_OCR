# PCB图像文本与元件匹配系统

本项目是一个基于深度学习的PCB（印刷电路板）图像分析系统，能够检测、识别并匹配PCB图像中的文本和电子元件。

## 功能特性

- **目标检测**：使用YOLO模型检测PCB图像中的电子元件
- **文本识别**：使用PaddleOCR识别检测到的文本区域
- **文本-元件匹配**：智能匹配文本与其对应的电子元件
- **语义分割**：对检测到的元件进行精确分割
- **结果可视化**：生成包含匹配关系的可视化图像

## 系统架构

```
输入图像 → YOLO检测 → 文本识别 → 元件分割 → 文本-元件匹配 → 结果可视化
```

### 主要模块

1. **[main.py](file:///home/wayen/image_pcb/image_extract/main.py)** - 主程序入口
2. **[utils/image_processor.py](file:///home/wayen/image_pcb/image_extract/utils/image_processor.py)** - 图像处理核心逻辑
3. **[utils/text_processor.py](file:///home/wayen/image_pcb/image_extract/utils/text_processor.py)** - 文本识别处理
4. **[utils/text_component_matcher.py](file:///home/wayen/image_pcb/image_extract/utils/text_component_matcher.py)** - 文本-元件匹配算法
5. **[utils/fasttext_classifier.py](file:///home/wayen/image_pcb/image_extract/utils/fasttext_classifier.py)** - FastText分类器

## 安装依赖

```bash
# 激活GPU环境
conda activate gpu

# 安装必要依赖
pip install ultralytics opencv-python numpy paddlepaddle paddlex
```

## 使用方法

### 命令行参数

```bash
python main.py --image <图像路径> --output <输出目录> --detection-classes <要移除的类别> --mode <运行模式>
```

### 参数说明

- `--image, -i`: 单张图像路径
- `--folder, -f`: 图像文件夹路径
- `--output, -o`: 输出目录路径 (默认: predict_res)
- `--detection-classes, -dc`: 要从检测结果中移除的类别ID列表 (默认: [1])
- `--segmentation-conf, -sc`: 分割置信度阈值 (默认: 0.5)
- `--mode, -m`: 运行模式: detect_and_segment(检测并分割) 或 detect_only(仅检测) (默认: detect_and_segment)

### 示例用法

```bash
# 处理单张图像
python main.py -i ./images/sample.jpg -o ./runs/run01 --mode detect_and_segment

# 处理图像文件夹
python main.py -f ./images/ -o ./runs/run02 --mode detect_and_segment
```

## 输出结构

```
<output_dir>/
├── detection_results/           # 检测结果
│   ├── connector/             # 连接器检测结果
│   ├── text/                  # 文本检测结果
│   └── classes/               # 分割类别结果
├── processed_images/          # 处理后的图像
├── original/                  # 原始图像
├── segmented/                 # 分割结果
├── debug/                     # 调试文件
├── text_results.txt           # 文本识别与匹配结果
└── matching_result.png        # 匹配可视化结果
```

## 文本-元件匹配算法

### 匹配策略

1. **芯片区域匹配**：在芯片分割轮廓内的文本直接匹配到芯片
2. **距离权重匹配**：基于文本与元件的几何距离进行匹配
3. **类别一致性匹配**：优先匹配相同类型的文本与元件
4. **匈牙利算法优化**：使用最优分配算法确保最佳匹配

### 匹配类别

- 芯片 (chip) - IC、微控制器等
- 电容 (capacitor) - 电容器件
- 电阻 (resistor) - 电阻器件
- 接地 (ground) - 接地点
- 连接器 (line_connector) - 连接点

## 可视化功能

### 图例布局

- **位置**：图像最左侧
- **排列**：上下居中对齐
- **标签**：使用英文标签确保兼容性
- **避免重叠**：图例与主图分离

### 颜色编码

- 芯片: 橙色 (0, 128, 255)
- 电容: 绿色 (0, 255, 0)
- 电阻: 红色 (255, 0, 0)
- 接地: 青色 (255, 255, 0)
- 连接器: 紫色 (128, 0, 128)
- 文本: 蓝色 (0, 0, 255)
- 匹配线: 洋红色 (255, 0, 255)

## 配置参数

### 匹配阈值

- 文本置信度阈值: 0.6
- 元件置信度阈值: 0.75
- 匹配得分阈值: 0.5

### 权重配置

- 距离权重: -0.7 (主要因素)
- 相对距离权重: -0.25
- 方向权重: 0.1
- IOU权重: 0.15
- 尺寸比权重: 0.05
- 文本置信度权重: 0.1
- 元件置信度权重: 0.1

## 文本预处理

### 扩大检测框

- 文本检测框扩大1.2倍以提高识别准确性
- 通过等比例缩放和居中放置优化文本识别

### 特殊字符处理

- μ → u
- Ω → ohm

## 调试功能

所有调试文件（以debug_开头）将自动保存到输出目录的debug子文件夹中，便于跟踪处理过程。

## 环境要求

- Python 3.8+
- CUDA兼容GPU（推荐）
- 至少8GB内存

## 许可证

[在此处添加许可证信息]