import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def apply_morphological_opening(image_path):
    """
    对指定图像进行开运算处理
    """
    # 读取原始图像
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    # 转换为灰度图
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 开运算处理
    kernel = np.ones((5, 5), np.uint8)  # 使用5x5的结构元素
    opened_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)

    # 使用matplotlib显示原始图像和开运算结果
    plt.figure(figsize=(12, 6))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image', fontsize=12)
    plt.axis('off')

    # 开运算结果
    plt.subplot(1, 2, 2)
    plt.imshow(opened_img, cmap='gray')
    plt.title('Morphological Opening Result', fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle('Morphological Opening Operation', fontsize=16, y=0.98)
    plt.show()

    # 同时也使用OpenCV窗口显示
    cv2.imshow('Original Image', original_img)
    cv2.imshow('Morphological Opening Result', opened_img)

    print("Displaying morphological opening results. Press any key or wait 10 seconds to close...")
    cv2.waitKey(10000)  # 等待10秒
    cv2.destroyAllWindows()

    # 保存处理后的图像
    output_path = image_path.parent / f"{image_path.stem}_opened.jpg"
    cv2.imwrite(str(output_path), opened_img)
    print(f"开运算处理后的图像已保存到: {output_path}")

def main():
    # 指定图像路径
    image_path = Path(r"C:\Users\11\Desktop\pj\image_extract\Power Monitor_TPS3840PL20DBVR V3.png")

    # 检查图像是否存在
    if not image_path.exists():
        print(f"错误: 图像文件不存在 {image_path}")
        return

    # 执行开运算处理
    apply_morphological_opening(image_path)

if __name__ == "__main__":
    main()