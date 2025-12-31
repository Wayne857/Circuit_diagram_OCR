# SegMAN Custom Dataset Training Guide

## Overview

This guide explains how to train the SegMAN model on your custom dataset. The SegMAN model is designed for semantic segmentation tasks and achieves state-of-the-art performance on various benchmarks.

## Prerequisites

1. **Environment Setup**: Make sure you have installed all dependencies as described in the main README.md
2. **Dataset Preparation**: Your dataset should be organized in the following structure:
   ```
   dataset_root/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

## Dataset Format

Your dataset should follow these conventions:
- Images should be in common formats (JPEG, PNG, etc.)
- Labels should be in the same format as your current dataset (text files with polygon coordinates)
- The label format should be compatible with MMSegmentation framework

## Configuration Files

We have prepared configuration files for your custom dataset:
1. `local_configs/_base_/datasets/custom_dataset.py` - Dataset configuration
2. `local_configs/segman/base/segman_b_custom.py` - Model and training configuration

## Training Steps

### 1. Prepare Pretrained Weights
Download the ImageNet-1k pretrained weights and place them in the `pretrained/` folder.

### 2. Modify Configuration (if needed)
Check and modify the following in `local_configs/segman/base/segman_b_custom.py`:
- `num_classes`: Set to match your dataset (currently set to 12)
- `pretrained`: Point to your pretrained encoder weights
- `data_root`: Verify the path to your dataset

### 3. Start Training
Run the training script:
```bash
# Single GPU training
python tools/train.py local_configs/segman/base/segman_b_custom.py --work-dir outputs/custom_dataset

# Multi-GPU training (if you have multiple GPUs)
bash tools/dist_train.sh local_configs/segman/base/segman_b_custom.py <GPU_NUM> --work-dir outputs/custom_dataset
```

Alternatively, you can use the provided script:
```bash
bash scripts/train_custom.sh
```

## Training Parameters

The default training configuration includes:
- Batch size: 4 samples per GPU
- Total batch size: 16 (with 4 GPUs)
- Learning rate: 0.00006
- Epochs: Defined in schedule configuration
- Optimizer: AdamW
- Scheduler: Polynomial decay

## Monitoring Training

During training, you can monitor:
- Loss values in the console output
- Checkpoints saved in the `work-dir` directory
- TensorBoard logs (if enabled) for visualization

## Evaluation

To evaluate the trained model:
```bash
python tools/test.py local_configs/segman/base/segman_b_custom.py /path/to/checkpoint_file
```

## Troubleshooting

### Common Issues

1. **Data Loading Issues**:
   - Verify that your dataset paths are correct
   - Ensure image and label filenames match (excluding extensions)
   - Check that label files are in the correct format

2. **Memory Issues**:
   - Reduce batch size in the configuration
   - Use smaller input image sizes
   - Enable gradient checkpointing

3. **Performance Issues**:
   - Check class imbalance in your dataset
   - Adjust loss weights if needed
   - Fine-tune hyperparameters

### Class Mapping

The current class mapping is:
0. resistor
1. motor
2. ground
3. line
4. arrow
5. line_connector
6. chip
7. capacitor
8. zener_diode
9. mov
10. fuse
11. inductor

If your dataset uses different class names or order, update the `classes` tuple in the dataset configuration file.

## Advanced Usage

### Customizing the Model
You can modify the model architecture by changing parameters in the configuration file:
- Backbone type and parameters
- Decoder settings
- Loss functions

### Data Augmentation
Adjust data augmentation strategies in the pipeline configuration to better suit your dataset characteristics.

## Conclusion

This guide provides a complete workflow for training SegMAN on your custom dataset. Follow the steps carefully and adjust configurations as needed for optimal performance.