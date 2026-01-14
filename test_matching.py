#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试文本与元件匹配功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from utils.text_component_matcher import TextComponentMatcher


def test_matching():
    """测试匹配功能"""
    print("测试文本与元件匹配功能...")
    
    # 测试数据
    text_data = [
        {'text': 'TPS3840D', 'coord': (107, 161, 213, 176), 'category': '芯片', 'conf': 0.5773},
        {'text': '10μF', 'coord': (16, 222, 53, 240), 'category': '电容', 'conf': 0.7862},
        {'text': 'RESET', 'coord': (179, 123, 226, 137), 'category': '芯片', 'conf': 0.9842},
    ]

    component_data = [
        {'category': 'capacitor', 'bbox': (127, 26, 145, 52), 'conf': 0.94, 'segmentation': [[135, 25], [135, 30], [135, 31], [135, 31], [134, 32], [127, 32], [127, 40], [134, 40], [135, 41], [135, 41]]},
        {'category': 'chip', 'bbox': (75, 62, 252, 250), 'conf': 0.83, 'segmentation': [[132, 78], [99, 78], [98, 78], [92, 78], [90, 80], [90, 87], [90, 88], [90, 129], [89, 129], [86, 129], [86, 130], [86, 240], [87, 240], [90, 242], [90, 245], [92, 248], [132, 248], [252, 248], [252, 78], [132, 78]]},
    ]

    matcher = TextComponentMatcher()
    matches, img_w, img_h = matcher.optimal_text_component_matching(text_data, component_data)

    print('匹配结果:')
    for idx, item in enumerate(matches):
        comp = item['component']
        texts = item['matched_texts']
        comp_info = f'[{idx+1}] 元件：{comp["category"]} | 检测框：{comp["bbox"]}'
        text_list = [t['text'] for t in texts] if texts else ['无匹配文本']
        print(f'{comp_info} | 匹配文本：{", ".join(text_list)}')
    
    # 测试格式化输出
    print("\n格式化输出:")
    formatted_result = matcher.format_match_results(matches)
    print(formatted_result)
    
    print(f"\n图片尺寸: {img_w}x{img_h}")
    print("测试完成！")


if __name__ == "__main__":
    test_matching()