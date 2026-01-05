"""
测试文本识别功能的简化脚本
"""
import numpy as np
import cv2
from utils.text_processor import TextProcessor
from pathlib import Path
import tempfile
import os

def test_text_recognition():
    print("初始化TextProcessor...")
    tp = TextProcessor()
    print(f"使用真实模型: {tp.use_real_model}")
    
    # 创建一个测试图像
    print("创建测试图像...")
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # 测试单个图像识别
    print("测试单个图像识别...")
    result = tp.recognize_text_in_cropped_image(test_img, 'test_image', (10, 10, 50, 50))
    print(f"单个图像识别结果: {result}")
    
    # 测试多个图像并发识别（使用较少的图像以减少时间）
    print("测试并发识别（仅2个图像）...")
    test_data = [
        {'image': test_img, 'name': 'test1', 'bbox': (10, 10, 50, 50)},
        {'image': test_img, 'name': 'test2', 'bbox': (20, 20, 60, 60)}
    ]
    
    results = tp.recognize_texts_concurrent(test_data, max_workers=2)
    print(f"并发识别完成，结果数量: {len(results)}")
    
    # 创建一个真实的图像文件进行测试
    print("测试真实图像文件...")
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, test_img)
        try:
            file_result = tp.recognize_text_in_image(tmp.name)
            print(f"文件识别结果: {file_result}")
        finally:
            os.unlink(tmp.name)

if __name__ == "__main__":
    test_text_recognition()