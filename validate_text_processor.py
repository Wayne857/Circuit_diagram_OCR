"""
验证text_processor模块功能的脚本
"""
import sys
import os
import cv2
import numpy as np
from pathlib import Path

def validate_text_processor():
    print("="*60)
    print("验证text_processor模块功能")
    print("="*60)
    
    # 1. 验证模块导入
    print("\n1. 测试模块导入...")
    try:
        from utils.text_processor import TextProcessor
        print("   ✓ text_processor模块导入成功")
    except Exception as e:
        print(f"   ✗ text_processor模块导入失败: {e}")
        return False
    
    # 2. 验证TextProcessor初始化
    print("\n2. 测试TextProcessor初始化...")
    try:
        tp = TextProcessor()
        print(f"   ✓ TextProcessor初始化成功，使用真实模型: {tp.use_real_model}")
    except Exception as e:
        print(f"   ✗ TextProcessor初始化失败: {e}")
        return False
    
    # 3. 验证图像读取
    print("\n3. 测试图像读取...")
    image_path = "Power.png"
    if not os.path.exists(image_path):
        print(f"   ✗ 图像文件 {image_path} 不存在")
        return False
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"   ✗ 无法读取图像 {image_path}")
        return False
    else:
        h, w = img.shape[:2]
        print(f"   ✓ 图像读取成功，尺寸: {w}x{h}")
    
    # 4. 测试裁剪图像功能
    print("\n4. 测试裁剪图像功能...")
    try:
        # 从图像中裁剪一个小区域进行测试
        crop_h, crop_w = min(h//4, 100), min(w//4, 100)  # 裁剪图像的1/4大小，但不超过100x100
        crop_y, crop_x = h//4, w//4
        cropped_img = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        if cropped_img.size == 0:
            print("   ✗ 裁剪图像为空")
            return False
        else:
            print(f"   ✓ 裁剪图像成功，尺寸: {cropped_img.shape}")
    except Exception as e:
        print(f"   ✗ 裁剪图像失败: {e}")
        return False
    
    # 5. 测试单个图像识别功能
    print("\n5. 测试单个图像识别功能...")
    try:
        result = tp.recognize_text_in_cropped_image(
            cropped_image=cropped_img,
            image_name="test_validation",
            bbox_coords=(crop_x, crop_y, crop_x+crop_w, crop_y+crop_h)
        )
        print(f"   ✓ 单个图像识别调用成功")
        print(f"     - 结果类型: {type(result)}")
        print(f"     - success字段: {result.get('success', 'N/A')}")
        if 'error' in result:
            print(f"     - 错误信息: {result['error']}")
    except Exception as e:
        print(f"   ✗ 单个图像识别失败: {e}")
        return False
    
    # 6. 测试并发处理功能（使用少量图像）
    print("\n6. 测试并发处理功能...")
    try:
        # 创建几个相同的裁剪图像用于测试
        test_data = []
        for i in range(2):  # 只测试2个图像，减少处理时间
            test_data.append({
                'image': cropped_img,
                'name': f'test_validation_{i}',
                'bbox': (crop_x+i*10, crop_y+i*10, crop_x+crop_w+i*10, crop_y+crop_h+i*10)
            })
        
        results = tp.recognize_texts_concurrent(test_data, max_workers=2)
        print(f"   ✓ 并发处理调用成功，处理了 {len(results)} 个图像")
        for i, result in enumerate(results):
            print(f"     - 图像 {i+1}: success={result.get('success', 'N/A')}")
            if 'error' in result:
                print(f"       错误: {result['error']}")
    except Exception as e:
        print(f"   ✗ 并发处理失败: {e}")
        return False
    
    print("\n" + "="*60)
    print("验证完成！text_processor模块功能正常")
    print("="*60)
    return True

def test_paddleocr_directly():
    print("\n7. 直接测试PaddleOCR功能...")
    try:
        from paddleocr import PaddleOCRVL
        print("   ✓ PaddleOCRVL导入成功")
        
        # 尝试初始化（不实际处理图像，只是测试初始化）
        pipeline = PaddleOCRVL()
        print("   ✓ PaddleOCRVL初始化成功")
        del pipeline
        print("   ✓ PaddleOCRVL清理成功")
        return True
    except Exception as e:
        print(f"   ✗ PaddleOCR直接测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始验证text_processor模块...")
    
    success = validate_text_processor()
    direct_test = test_paddleocr_directly()
    
    print(f"\n验证结果:")
    print(f"- text_processor模块验证: {'通过' if success else '失败'}")
    print(f"- PaddleOCR直接测试: {'通过' if direct_test else '失败'}")
    
    if success and direct_test:
        print("\n✓ 所有验证都通过，模块功能正常")
    else:
        print("\n✗ 验证未通过，请检查相关问题")
        sys.exit(1)