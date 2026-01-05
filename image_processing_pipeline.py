import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def image_processing_pipeline(image_path):
    """
    图像处理流水线：
    1. 大津法二值化
    2. 图像金字塔向上取样
    3. 先腐蚀再膨胀的开环运算
    """
    # 读取原始图像
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return
    
    # 转换为灰度图以进行二值化
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 步骤1: 大津法二值化
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 步骤2: 图像金字塔向上取样
    upsampled_img = cv2.pyrUp(binary_img)
    
    # 步骤3: 先腐蚀再膨胀的开环运算
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)
    
    # 先腐蚀
    eroded_img = cv2.erode(upsampled_img, kernel, iterations=1)
    
    # 再膨胀
    processed_img = cv2.dilate(eroded_img, kernel, iterations=1)
    
    # 使用matplotlib显示所有图像
    plt.figure(figsize=(16, 12))
    
    # 原始图像
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    
    # 灰度图像
    plt.subplot(2, 3, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('Grayscale Image', fontsize=12)
    plt.axis('off')
    
    # 二值化图像
    plt.subplot(2, 3, 3)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Otsu Binarization', fontsize=12)
    plt.axis('off')
    
    # 金字塔向上取样
    plt.subplot(2, 3, 4)
    plt.imshow(upsampled_img, cmap='gray')
    plt.title('Pyramid Up Sampling', fontsize=12)
    plt.axis('off')
    
    # 腐蚀后的图像
    plt.subplot(2, 3, 5)
    plt.imshow(eroded_img, cmap='gray')
    plt.title('Erosion Operation', fontsize=12)
    plt.axis('off')
    
    # 最终处理结果（开环运算）
    plt.subplot(2, 3, 6)
    plt.imshow(processed_img, cmap='gray')
    plt.title('Opening Operation (Erosion + Dilation)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Image Processing Pipeline', fontsize=16, y=0.98)
    plt.show()
    
    # 同时也使用OpenCV窗口显示
    cv2.imshow('原始图像', original_img)
    cv2.imshow('灰度图像', gray_img)
    cv2.imshow('大津法二值化', binary_img)
    cv2.imshow('图像金字塔向上取样', upsampled_img)
    cv2.imshow('腐蚀操作', eroded_img)
    cv2.imshow('开环运算(腐蚀+膨胀)', processed_img)
    
    print("显示图像处理结果，按任意键或等待20秒后自动关闭...")
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