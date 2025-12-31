#!/bin/bash

# Custom dataset training script for SegMAN-B

# Single-gpu training
python tools/train.py local_configs/segman/base/segman_b_custom.py --work-dir outputs/custom_dataset

# Multi-gpu training (uncomment if you have multiple GPUs)
# bash tools/dist_train.sh local_configs/segman/base/segman_b_custom.py 2 --work-dir outputs/custom_dataset