#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
背景色检测工具
用于检测图像中的背景色，参考proc_merge.py中的直方图分析方法
"""

import cv2
import numpy as np
from collections import Counter


def get_background_color(image):
    """
    从图像中检测背景色
    参考proc_merge.py中的直方图分析方法
    
    Args:
        image: 输入图像 (numpy array)
        
    Returns:
        tuple: 背景色 (B, G, R) 或灰度值
    """
    if image is None:
        raise ValueError("输入图像不能为空")
    
    # 转换为灰度图进行分析
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # 找到最高峰（通常是背景色）
    bg_gray = int(np.argmax(hist))
    
    # 如果图像是彩色的，返回彩色背景色
    if len(image.shape) == 3:
        # 方法1：使用直方图峰值附近的像素值作为背景色
        # 创建掩码，找到接近背景灰度值的像素
        mask = np.abs(gray.astype(int) - bg_gray) <= 5  # 容差为5
        
        # 获取这些像素的平均颜色
        if np.any(mask):
            bg_bgr = np.mean(image[mask], axis=0).astype(int)
            return tuple(bg_bgr)
        else:
            # 如果没有找到匹配的像素，使用最频繁的颜色
            reshaped = image.reshape(-1, 3)
            # 获取最常见的颜色
            pixel_tuples = [tuple(pixel) for pixel in reshaped]
            counter = Counter(pixel_tuples)
            most_common_bgr = counter.most_common(1)[0][0]
            return most_common_bgr
    else:
        return int(bg_gray)


def get_background_color_advanced(image):
    """
    更高级的背景色检测方法
    使用多种策略确保准确识别背景色
    
    Args:
        image: 输入图像 (numpy array)
        
    Returns:
        tuple: 背景色 (B, G, R) 或灰度值
    """
    if image is None:
        raise ValueError("输入图像不能为空")
    
    # 转换为灰度图进行分析
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    
    # 找到最高峰（通常是背景色）
    bg_gray = int(np.argmax(hist))
    
    # 如果图像是彩色的，返回彩色背景色
    if len(image.shape) == 3:
        # 使用更稳健的方法获取背景色
        # 在直方图峰值附近的较大范围内寻找代表性颜色
        lower_bound = max(0, bg_gray - 10)
        upper_bound = min(255, bg_gray + 10)
        
        # 创建掩码，找到在背景灰度值范围内的像素
        mask = (gray >= lower_bound) & (gray <= upper_bound)
        
        if np.any(mask):
            # 获取这些像素的平均颜色
            bg_bgr = np.mean(image[mask], axis=0).astype(int)
            return tuple(bg_bgr)
        else:
            # 回退到最常见颜色
            reshaped = image.reshape(-1, 3)
            pixel_tuples = [tuple(pixel) for pixel in reshaped]
            counter = Counter(pixel_tuples)
            most_common_bgr = counter.most_common(1)[0][0]
            return most_common_bgr
    else:
        return int(bg_gray)


if __name__ == "__main__":
    # 测试函数
    import sys
    from pathlib import Path
    
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # 示例用法
    # image = cv2.imread("your_image.jpg")
    # bg_color = get_background_color(image)
    # print(f"检测到的背景色: {bg_color}")
    pass