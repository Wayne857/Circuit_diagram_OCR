# 图像处理流水线脚本说明

## 概述

本项目包含一个图像处理流水线脚本，对指定图像执行以下处理步骤：
1. CLAHE（限制对比度自适应直方图均衡化）
2. 开运算 (Opening Operation)
3. 图像锐化 (Image Sharpening)
4. 手动二值化 (Manual Binarization)
5. 图像金字塔向上取样 (Pyramid Up Sampling)
6. 开运算 (Opening Operation)

## 文件说明

- `image_processing_pipeline_en.py` - 主要的图像处理脚本（英文界面）
- `IMAGE_PROCESSING_PIPELINE_README.md` - 本说明文档

## 功能特点

### 1. CLAHE（限制对比度自适应直方图均衡化）
- 限制对比度以避免过度增强
- 适合处理局部光照不均的图像
- tileGridSize=(8,8) 分块处理

### 2. 开运算
- 使用morphologyEx函数实现
- 用于去除小的噪声点
- 保持图像主要形状特征

### 3. 图像锐化
- 使用3x3锐化核
- 增强边缘清晰度
- 核心权重为5，周围为-1

### 4. 手动二值化
- 阈值设为70
- 灰度值大于70的像素变为255
- 灰度值小于等于70的像素变为0

### 5. 图像金字塔向上取样
- 增加图像分辨率
- 使用高斯核进行插值
- 图像尺寸增加一倍

### 6. 开运算
- 使用morphologyEx函数实现
- 用于去除小的噪声点
- 保持图像主要形状特征

## 使用方法

```bash
python image_processing_pipeline_en.py
```

脚本将自动处理指定的图像文件：
`imagessegment/seg/images/val/AC-DC Voltage Conversion_LS03-13B12R3 V3.png`

## 输出结果

脚本将显示以下8个步骤的图像：
1. 原始图像
2. 灰度图像
3. CLAHE处理结果
4. 开运算结果
5. 锐化处理结果
6. 手动二值化结果
7. 金字塔向上取样结果
8. 开运算最终结果

## 显示方式

- 使用matplotlib显示所有图像的组合视图
- 使用OpenCV分别显示每个处理步骤的图像
- 所有图像同时显示，等待20秒后自动关闭

## 技术细节

- 使用3x3的结构元素进行形态学操作
- CLAHE使用clipLimit=2.0, tileGridSize=(8,8)
- 手动二值化使用阈值70
- 开运算使用cv2.morphologyEx函数的cv2.MORPH_OPEN参数
- 图像金字塔使用高斯核进行上采样

## 依赖库

- OpenCV
- NumPy
- Matplotlib
- pathlib

## 自定义

如需处理其他图像，可修改脚本中的图像路径：
```python
image_path = Path("your/image/path.png")
```

该脚本实现了完整的图像处理流水线，可用于电路图等二值化处理和形态学分析。