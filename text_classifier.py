#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本分类脚本
使用训练好的FastText模型对输入的文本进行电路元件分类
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from utils.fasttext_classifier import FastTextComponentClassifier


def classify_single_text(text: str, model_path: str = "component_fasttext.bin"):
    """
    对单个文本进行分类
    
    Args:
        text (str): 待分类的文本
        model_path (str): 模型文件路径
    
    Returns:
        dict: 分类结果字典
    """
    try:
        # 初始化分类器
        classifier = FastTextComponentClassifier(model_path)
        
        # 进行预测
        result = classifier.predict(text)
        
        return result
    except Exception as e:
        print(f"分类过程中出现错误: {e}")
        return None


def classify_multiple_texts(texts: list, model_path: str = "component_fasttext.bin"):
    """
    对多个文本进行批量分类
    
    Args:
        texts (list): 待分类的文本列表
        model_path (str): 模型文件路径
    
    Returns:
        list: 分类结果列表
    """
    try:
        # 初始化分类器
        classifier = FastTextComponentClassifier(model_path)
        
        # 进行批量预测
        results = classifier.predict_batch(texts)
        
        return results
    except Exception as e:
        print(f"批量分类过程中出现错误: {e}")
        return None


def main():
    """主函数，提供命令行交互界面"""
    print("电路元件文本分类器")
    print("=" * 40)
    
    # 检查模型文件是否存在
    model_path = "component_fasttext.bin"
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        print("请确保训练好的FastText模型文件存在于项目根目录中")
        return
    
    print("模型加载成功！")
    print("输入 'quit' 或 'exit' 退出程序")
    print("-" * 40)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入要分类的文本: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("程序已退出")
                break
            
            if not user_input:
                print("输入不能为空，请重新输入")
                continue
            
            # 对输入文本进行分类
            result = classify_single_text(user_input, model_path)
            
            if result:
                print(f"输入文本: {result['text']}")
                print(f"预测类别: {result['predicted_label']}")
                print(f"置信度: {result['confidence']}")
            else:
                print("分类失败")
                
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"处理过程中出现错误: {e}")


def batch_classify_from_file(file_path: str, output_path: str = None, model_path: str = "component_fasttext.bin"):
    """
    从文件中读取文本进行批量分类
    
    Args:
        file_path (str): 输入文件路径，每行一个文本
        output_path (str): 输出文件路径，如果为None则打印到控制台
        model_path (str): 模型文件路径
    """
    try:
        # 读取输入文件
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"从文件 {file_path} 读取了 {len(texts)} 个文本")
        
        # 批量分类
        results = classify_multiple_texts(texts, model_path)
        
        if results is None:
            print("批量分类失败")
            return
        
        # 输出结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("文本分类结果:\n")
                f.write("-" * 50 + "\n")
                for i, result in enumerate(results):
                    f.write(f"文本{i+1}: {result['text']}\n")
                    f.write(f"类别: {result['predicted_label']}\n")
                    f.write(f"置信度: {result['confidence']}\n")
                    f.write("-" * 30 + "\n")
            print(f"分类结果已保存到: {output_path}")
        else:
            print("\n文本分类结果:")
            print("-" * 50)
            for i, result in enumerate(results):
                print(f"文本{i+1}: {result['text']}")
                print(f"类别: {result['predicted_label']}")
                print(f"置信度: {result['confidence']}")
                print("-" * 30)
                
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {file_path}")
    except Exception as e:
        print(f"批量分类过程中出现错误: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='电路元件文本分类器')
    parser.add_argument('input', nargs='?', help='输入文本或输入文件路径')
    parser.add_argument('-f', '--file', action='store_true', help='输入为文件路径')
    parser.add_argument('-o', '--output', type=str, help='输出文件路径（仅在文件模式下有效）')
    parser.add_argument('-m', '--model', type=str, default='component_fasttext.bin', help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.file:
        # 文件模式
        if not args.input:
            print("错误: 使用文件模式需要指定输入文件路径")
            sys.exit(1)
        batch_classify_from_file(args.input, args.output, args.model)
    elif args.input:
        # 单文本模式
        result = classify_single_text(args.input, args.model)
        if result:
            print(f"输入文本: {result['text']}")
            print(f"预测类别: {result['predicted_label']}")
            print(f"置信度: {result['confidence']}")
        else:
            print("分类失败")
    else:
        # 交互模式
        main()