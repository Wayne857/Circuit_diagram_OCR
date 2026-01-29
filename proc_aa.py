import cv2
import numpy as np
import matplotlib.pyplot as plt

def merge_lines_morphology_show(image_path, max_gap=3):
    """
    使用形态学闭运算合并断裂线段，并用matplotlib显示（无Qt依赖）
    """
    # 1. 读取图像（不使用GUI）
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {image_path}")
    
    # 2. 预处理：灰度 + 二值化（PCB铜线通常为亮色，背景暗）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 移除 _INV 避免反色
    
    # 3. 闭运算连接缝隙（自适应核大小）
    kernel_size = max_gap * 2 + 1
    # 使用十字核适应多方向线段（PCB中常见水平/垂直/45°线）
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 4. 用matplotlib显示（无Qt依赖）
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title('Original (RGB)')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('Binary (Before Merge)')
    plt.imshow(binary, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f'Merged (Gap≤{max_gap}px)')
    plt.imshow(closed, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return closed  # 返回合并后的二值图供后续处理

# 直接运行（无需保存）
try:
    merged_binary = merge_lines_morphology_show('imgs/image.png', max_gap=3)
    print("✅ 合并完成！左侧:原图 | 中间:断裂线段 | 右侧:合并后结果")
    
    # 可选：统计合并效果（对PCB工程有价值）
    num_before = cv2.connectedComponents(binary)[0] - 1
    num_after = cv2.connectedComponents(merged_binary)[0] - 1
    print(f"📊 线段数量变化: {num_before} → {num_after} (减少 {num_before-num_after} 条)")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    print("💡 建议: 检查 imgs/image.png 是否存在，或尝试调整 max_gap (3~7)")