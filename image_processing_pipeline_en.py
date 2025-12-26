import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def image_processing_pipeline(image_path):
    """
    图像处理流水线：
    1. CLAHE（限制对比度自适应直方图均衡化）
    2. 开运算
    3. 图像锐化
    4. 手动二值化（阈值70）
    5. 图像金字塔向上取样
    6. 开运算
    """
    # 读取原始图像
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 转换为灰度图以进行二值化
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 步骤1: 使用CLAHE（限制对比度自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # clipLimit控制对比度
    img_clahe = clahe.apply(gray_img)
    
    # 步骤2: 开运算
    kernel_open = np.ones((3, 3), np.uint8)
    img_open = cv2.morphologyEx(img_clahe, cv2.MORPH_OPEN, kernel_open)
    
    # 步骤3: 锐化处理
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    img_sharpen = cv2.filter2D(img_open, -1, kernel_sharpen)
    
    # 步骤4: 手动二值化 - 灰度值大于70的都变为255
    _, binary_img = cv2.threshold(img_sharpen, 70, 255, cv2.THRESH_BINARY)
    
    # 步骤3: 图像金字塔向上取样
    upsampled_img = cv2.pyrUp(binary_img)
    
    # 步骤4: 开运算（使用morphologyEx函数）
    # 定义结构元素
    kernel = np.ones((3, 3), np.uint8)
    
    # 使用morphologyEx函数进行开运算
    processed_img = cv2.morphologyEx(upsampled_img, cv2.MORPH_OPEN, kernel)
    
    # 使用matplotlib显示所有图像
    plt.figure(figsize=(15, 22))  # 使用4x2布局以显示所有8个步骤
    
    # 原始图像
    plt.subplot(4, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    # 灰度图像
    plt.subplot(4, 2, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image', fontsize=12)
    plt.axis('off')
    
    # CLAHE处理后图像
    plt.subplot(4, 2, 3)
    plt.imshow(img_clahe, cmap='gray')
    plt.title('CLAHE', fontsize=12)
    plt.axis('off')
    
    # 开运算后图像
    plt.subplot(4, 2, 4)
    plt.imshow(img_open, cmap='gray')
    plt.title('Opening Operation', fontsize=12)
    plt.axis('off')
    
    # 锐化处理后图像
    plt.subplot(4, 2, 5)
    plt.imshow(img_sharpen, cmap='gray')
    plt.title('Sharpening', fontsize=12)
    plt.axis('off')
    
    # 二值化图像
    plt.subplot(4, 2, 6)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Manual Binarization (Thresh=70)', fontsize=12)
    plt.axis('off')
    
    # 金字塔向上取样
    plt.subplot(4, 2, 7)
    plt.imshow(upsampled_img, cmap='gray')
    plt.title('Pyramid Up Sampling', fontsize=12)
    plt.axis('off')
    
    # 最终开运算结果
    plt.subplot(4, 2, 8)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Final Opening Operation', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Image Processing Pipeline', fontsize=16, y=0.98)
    plt.show()
    
    # 同时也使用OpenCV窗口显示
    cv2.imshow('Original Image', original_img)
    cv2.imshow('Grayscale Image', gray_img)
    cv2.imshow('CLAHE', img_clahe)
    cv2.imshow('Opening Operation (Before Sharpen)', img_open)
    cv2.imshow('Sharpening', img_sharpen)
    cv2.imshow('Manual Binarization (Thresh=70)', binary_img)
    cv2.imshow('Pyramid Up Sampling', upsampled_img)
    cv2.imshow('Opening Operation (Final)', processed_img)
    
    print("Displaying image processing results. Press any key or wait 20 seconds to close...")
    cv2.waitKey(20000)  # 等待20秒
    cv2.destroyAllWindows()

def main():
    # 指定图像路径
    image_path = Path("imagessegment/seg/images/val/AC-DC Voltage Conversion_LS03-13B12R3 V3.png")
    
    # 检查图像是否存在
    if not image_path.exists():
        print(f"错误: 图像文件不存在 {image_path}")
        return
    
    # 执行图像处理流水线
    image_processing_pipeline(image_path)

if __name__ == "__main__":
    main()