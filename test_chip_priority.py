#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试芯片区域内文本优先匹配的功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from utils.text_component_matcher import TextComponentMatcher


def test_chip_priority():
    """测试芯片区域内文本优先匹配的功能"""
    print("测试芯片区域内文本优先匹配的功能...")
    
    # 创建一个测试用例：一个文本在芯片区域内，但类别是电容
    text_data = [
        {'text': 'CT', 'coord': (100, 100, 120, 120), 'category': '电容', 'conf': 0.68},  # 在芯片区域内
        {'text': 'NC', 'coord': (80, 80, 100, 100), 'category': '芯片', 'conf': 0.65},   # 在芯片区域内
        {'text': '10μF', 'coord': (200, 200, 220, 220), 'category': '电容', 'conf': 0.7862},  # 在电容器附近
    ]

    # 创建一个芯片，包含文本'CT'和'NC'
    component_data = [
        {'category': 'capacitor', 'bbox': (190, 190, 230, 230), 'conf': 0.92, 'segmentation': [[195, 195], [225, 195], [225, 225], [195, 225]]},
        {'category': 'chip', 'bbox': (50, 50, 150, 150), 'conf': 0.83, 'segmentation': [
            [60, 60], [70, 60], [80, 60], [90, 60], [100, 60], [110, 60], [120, 60], [130, 60], [140, 60], [140, 70], 
            [140, 80], [140, 90], [140, 100], [140, 110], [140, 120], [140, 130], [140, 140], [130, 140], [120, 140], 
            [110, 140], [100, 140], [90, 140], [80, 140], [70, 140], [60, 140], [60, 130], [60, 120], [60, 110], 
            [60, 100], [60, 90], [60, 80], [60, 70]
        ]},  # 定义一个包含点(100,100)和(80,80)的芯片区域
    ]

    matcher = TextComponentMatcher()
    matches, img_w, img_h = matcher.optimal_text_component_matching(text_data, component_data)

    print('芯片区域内文本优先匹配结果:')
    for idx, item in enumerate(matches):
        comp = item['component']
        texts = item['matched_texts']
        comp_info = f'[{idx+1}] 元件：{comp["category"]} | 检测框：{comp["bbox"]}'
        text_list = [t['text'] for t in texts] if texts else ['无匹配文本']
        print(f'{comp_info} | 匹配文本：{", ".join(text_list)}')
    
    # 验证：芯片应该匹配到'CT'和'NC'，即使'CT'的类别是'电容'
    chip_match = None
    for item in matches:
        if item['component']['category'] == 'chip':
            chip_match = item
            break
    
    if chip_match:
        chip_texts = [t['text'] for t in chip_match['matched_texts']]
        print(f"\n芯片匹配的文本: {chip_texts}")
        if 'CT' in chip_texts:
            print("✅ 验证通过：即使'CT'的类别是'电容'，但由于它在芯片区域内，仍然被正确匹配到芯片")
        else:
            print("❌ 验证失败：'CT'没有被匹配到芯片")
    
    print(f"\n图片尺寸: {img_w}x{img_h}")
    print("测试完成！")


if __name__ == "__main__":
    test_chip_priority()