# 使用SegMAN进行电路元件语义分割

本项目展示了如何将YOLO格式的电路元件数据集转换为语义分割格式，并使用SegMAN模型进行训练。

## 项目结构

```
image_extract/
├── imagessegment/seg/          # 原始YOLO格式数据集
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── segment.yaml
├── segman_dataset_fixed/       # 转换后的语义分割数据集（修复中文文件名问题）
├── segman_dataset/             # 转换后的语义分割数据集（含中文文件名乱码问题）
│   ├── img_dir/
│   │   ├── train/
│   │   └── val/
│   ├── ann_dir/
│   │   ├── train/
│   │   └── val/
│   └── classes.txt
├── SegMAN/                     # SegMAN项目
├── yolo_to_seg_converter.py    # YOLO到语义分割转换脚本
├── segman_config.py            # 模型配置文件
└── start_segman_training.py    # 训练启动脚本
```

## 数据集转换

原始数据集是YOLO格式的实例分割数据集，包含12个电路元件类别：

0. resistor (电阻)
1. motor (电机)
2. ground (地线)
3. line (连线)
4. arrow (箭头)
5. line_connector (连接器)
6. chip (芯片)
7. capacitor (电容)
8. zener_diode (齐纳二极管)
9. mov (压敏电阻)
10. fuse (保险丝)
11. inductor (电感)

转换脚本 `yolo_to_seg_converter.py` 将YOLO格式的多边形标签转换为语义分割所需的PNG格式标签图像。

## 配置文件

`SegMAN/segmentation/local_configs/custom/segman_custom.py` 是为您的数据集定制的配置文件，配置了：

- 12个类别的电路元件
- 适合的图像增强和预处理流水线
- AdamW优化器和学习率调度
- 每8000次迭代评估一次模型

## 训练流程

1. **数据预处理**：
   - 运行 `yolo_to_seg_converter.py` 脚本转换数据格式
   - 验证转换后的数据集结构

2. **环境设置**：
   - 安装所需依赖 (PyTorch, MMSegmentation, NATTEN等)
   - 确保SegMAN项目正确安装

3. **模型训练**：
   - 运行 `start_segman_training.py` 启动训练
   - 或直接在SegMAN目录下运行:
   ```bash
   cd SegMAN/segmentation
   python tools/train.py local_configs/custom/segman_custom.py --work-dir outputs/segman_custom
   ```

## 模型评估

训练过程中会定期评估模型性能，使用mIoU (mean Intersection over Union) 作为主要评估指标。

## 结果

训练完成后，模型权重将保存在 `outputs/segman_custom` 目录下，可以用于推理和测试。

## 注意事项

1. 确保有足够的GPU内存来运行训练
2. 根据硬件条件可适当调整batch size
3. 如需使用预训练权重，需要下载并更新配置文件中的路径
4. 训练时间取决于数据集大小和硬件性能，可能需要数小时到数天

## 技术支持

如遇到问题，请检查：
- 数据集路径是否正确
- 依赖库是否完整安装
- GPU驱动和CUDA版本是否兼容