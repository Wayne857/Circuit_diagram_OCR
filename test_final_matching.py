#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试最终优化后的文本与元件匹配功能
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from utils.text_component_matcher import TextComponentMatcher


def test_final_matching():
    """测试最终优化后的匹配功能"""
    print("测试最终优化后的文本与元件匹配功能...")
    
    # 模拟原始text_results.txt中的情况
    text_data = [
        {'text': 'TPS3840D', 'coord': (107, 161, 213, 176), 'category': '芯片', 'conf': 0.85},
        {'text': '10μF', 'coord': (16, 222, 53, 240), 'category': '电容', 'conf': 0.7862},
        {'text': 'NC', 'coord': (50, 100, 70, 120), 'category': '芯片', 'conf': 0.65},  # 位于电容器附近
        {'text': 'CT', 'coord': (60, 110, 80, 130), 'category': '电容', 'conf': 0.68},  # 位于电容器附近
        {'text': 'RESET', 'coord': (179, 123, 226, 137), 'category': '芯片', 'conf': 0.9842},
        {'text': 'MR', 'coord': (200, 50, 220, 70), 'category': '芯片', 'conf': 0.75},
        {'text': '100kΩ', 'coord': (199, 39, 246, 53), 'category': '电阻', 'conf': 0.7034},
        {'text': 'VDD', 'coord': (150, 100, 180, 120), 'category': '芯片', 'conf': 0.85},
        {'text': 'GND', 'coord': (189, 215, 224, 230), 'category': '接地', 'conf': 0.9948},
    ]

    component_data = [
        {'category': 'capacitor', 'bbox': (127, 26, 145, 52), 'conf': 0.94, 'segmentation': [[135, 25], [135, 30], [135, 31], [135, 31], [134, 32], [127, 32], [127, 40], [134, 40], [135, 41], [135, 41]]},
        {'category': 'capacitor', 'bbox': (59, 219, 77, 244), 'conf': 0.92, 'segmentation': [[67, 219], [67, 225], [66, 226], [59, 226], [59, 234], [65, 234], [67, 235], [67, 245], [69, 245], [69, 235]]},
        {'category': 'ground', 'bbox': (119, 54, 155, 73), 'conf': 0.88, 'segmentation': [[119, 54], [119, 58], [120, 58], [121, 59], [121, 59], [121, 59], [121, 59], [122, 59], [122, 60], [122, 60]]},
        {'category': 'line_connector', 'bbox': (255, 126, 265, 136), 'conf': 0.82, 'segmentation': [[256, 126], [264, 126], [264, 135], [256, 135]]},
        {'category': 'ground', 'bbox': (50, 262, 87, 280), 'conf': 0.78, 'segmentation': [[51, 263], [86, 263], [86, 279], [51, 279]]},
        {'category': 'resistor', 'bbox': (250, 0, 270, 68), 'conf': 0.84, 'segmentation': [[259, 0], [259, 28], [259, 29], [259, 30], [259, 30], [260, 30], [260, 31], [261, 31], [262, 31], [262, 31]]},
        {'category': 'chip', 'bbox': (75, 62, 252, 250), 'conf': 0.83, 'segmentation': [[132, 78], [99, 78], [98, 78], [92, 78], [90, 80], [90, 87], [90, 88], [90, 129], [89, 129], [86, 129], [86, 130], [86, 240], [87, 240], [90, 242], [90, 245], [92, 248], [132, 248], [252, 248], [252, 78], [132, 78]]},
        {'category': 'ground', 'bbox': (188, 252, 226, 270), 'conf': 0.86, 'segmentation': [[189, 253], [225, 253], [225, 269], [189, 269]]},
    ]

    matcher = TextComponentMatcher()
    matches, img_w, img_h = matcher.optimal_text_component_matching(text_data, component_data)

    print('按内容分类后匹配结果:')
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
    test_final_matching()