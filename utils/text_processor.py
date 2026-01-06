import random

# 尝试导入PaddleOCR，如果失败则使用模拟模式
try:
    from paddleocr import PaddleOCRVL
    # 尝试初始化PaddleOCRVL来确认它能正常工作
    test_pipeline = PaddleOCRVL()
    del test_pipeline
    PADDLEOCR_AVAILABLE = True
    print("PaddleOCRVL 可用")
except Exception as e:
    PADDLEOCR_AVAILABLE = False
    print(f"警告: 无法使用PaddleOCRVL ({e})，将使用模拟模式")
import cv2
import numpy as np
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any


class TextProcessor:
    """文本处理类，用于使用PaddleVL识别图像中的文本"""
    
    def __init__(self):
        """初始化文本处理器"""
        if PADDLEOCR_AVAILABLE:
            # 初始化PaddleOCRVL模型
            try:
                self.pipeline = PaddleOCRVL()
                self.use_real_model = True
                print("TextProcessor initialized with real PaddleOCRVL model")
            except Exception as e:
                print(f"警告: 无法初始化PaddleOCRVL模型 ({e})，将使用模拟模式")
                self.use_real_model = False
        else:
            # 使用模拟模式
            self.use_real_model = False
            print("TextProcessor initialized (simulated mode)")
        
    def recognize_text_in_image(self, image_path: str) -> Dict[str, Any]:
        """
        识别图像中的文本
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            Dict[str, Any]: 识别结果
        """
        try:
            if self.use_real_model:
                # 使用真实PaddleOCR模型
                print(f"使用真实模型处理: {image_path}")
                try:
                    import threading
                    import time
                    
                    result_container = [None]
                    exception_container = [None]
                    
                    def run_prediction():
                        try:
                            output = self.pipeline.predict(image_path)
                            result_container[0] = output
                        except Exception as e:
                            exception_container[0] = e
                    
                    # 创建线程运行预测
                    prediction_thread = threading.Thread(target=run_prediction)
                    prediction_thread.start()
                    prediction_thread.join(timeout=180)  # 增加超时到3分钟，以适应PaddleOCR处理时间
                    
                    if prediction_thread.is_alive():
                        # 如果线程仍在运行，说明超时了
                        print(f"真实模型处理超时: {image_path}")
                        # 返回一个基本结果而不是抛出异常，确保程序继续执行
                        return {
                            'success': False,
                            'data': None,
                            'image_path': image_path,
                            'error': 'Model timeout'
                        }
                    elif exception_container[0]:
                        # 如果有异常
                        print(f"真实模型处理异常: {exception_container[0]}")
                        # 返回错误结果而不是抛出异常，确保程序继续执行
                        return {
                            'success': False,
                            'data': None,
                            'image_path': image_path,
                            'error': str(exception_container[0])
                        }
                    else:
                        # 正常完成
                        output = result_container[0]
                        print(f"真实模型处理完成: {image_path}")
                        if output and len(output) > 0:
                            result = output[0]
                            return {
                                'success': True,
                                'data': result,
                                'image_path': image_path
                            }
                        else:
                            return {
                                'success': False,
                                'data': None,
                                'image_path': image_path,
                                'error': 'No text detected'
                            }
                except Exception as e:
                    print(f"真实模型处理出错: {e}")
                    # 返回错误结果而不是抛出异常，确保程序继续执行
                    return {
                        'success': False,
                        'data': None,
                        'image_path': image_path,
                        'error': str(e)
                    }
            else:
                # 模拟文本识别结果
                print(f"使用模拟模式处理: {image_path}")
                # 这里可以基于图像路径或其他因素生成模拟结果
                simulated_text = f"模拟识别结果 for {Path(image_path).name}"
                result = {
                    'parsing_res_list': [
                        {'text': simulated_text, 'score': random.uniform(0.8, 0.99)}
                    ]
                }
                
                return {
                    'success': True,
                    'data': result,
                    'image_path': image_path
                }
        except Exception as e:
            return {
                'success': False,
                'data': None,
                'image_path': image_path,
                'error': str(e)
            }
    
    def recognize_text_in_cropped_image(self, cropped_image, image_name: str, 
                                       bbox_coords: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """
        识别裁剪图像中的文本
        
        Args:
            cropped_image: 裁剪后的图像
            image_name (str): 图像名称
            bbox_coords (Tuple[int, int, int, int]): 检测框坐标 (x1, y1, x2, y2)
            
        Returns:
            Dict[str, Any]: 识别结果
        """
        # 临时保存裁剪图像到内存
        import tempfile
        import os
        
        temp_path = None  # 初始化为None，防止异常时未定义
        
        try:
            print(f"正在处理图像: {image_name}, 坐标: {bbox_coords}")
            # 验证裁剪图像是否有效
            if cropped_image is None or cropped_image.size == 0:
                raise ValueError(f"裁剪图像无效: {image_name}")
            
            # 图像预处理：如果最短边小于200，则等比放大至最短边为200
            h, w = cropped_image.shape[:2]
            min_side = min(h, w)
            
            if min_side < 200:
                scale = 200 / min_side
                new_h = int(h * scale)
                new_w = int(w * scale)
                
                # 使用cv2进行等比放大
                resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                print(f"图像已预处理: 原始尺寸 ({w}x{h}) -> 处理后尺寸 ({new_w}x{new_h})")
                
                # 创建1000x1000的空白图片，将放大后的图像放置在正中央
                final_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
                final_image.fill(255)  # 填充白色背景
                
                # 计算居中位置，但确保不会超出画布边界
                start_y = max(0, (1000 - new_h) // 2)
                start_x = max(0, (1000 - new_w) // 2)
                
                # 计算实际放置位置和尺寸
                end_y = min(start_y + new_h, 1000)
                end_x = min(start_x + new_w, 1000)
                
                # 确保不超过resized_image的实际尺寸
                actual_h = min(new_h, resized_image.shape[0])
                actual_w = min(new_w, resized_image.shape[1])
                
                # 将放大后的图像放置在中央，确保尺寸匹配
                paste_h = end_y - start_y
                paste_w = end_x - start_x
                
                if paste_h > 0 and paste_w > 0:
                    # 取较小的尺寸以避免形状不匹配
                    src_h = min(paste_h, actual_h)
                    src_w = min(paste_w, actual_w)
                    final_image[start_y:start_y+src_h, start_x:start_x+src_w] = resized_image[0:src_h, 0:src_w]
                
                processed_image = final_image
                print(f"图像已放置在1000x1000画布中央: 位置 ({start_x}, {start_y})")
                
                # 保存预处理后的图像用于调试
                debug_output_path = f"debug_preprocessed_{image_name}.jpg"
                cv2.imwrite(debug_output_path, processed_image)
                print(f"预处理图像已保存至: {debug_output_path}")
            else:
                # 即使图像尺寸 >= 200，仍然放置在2倍尺寸的空白背景中央
                new_h, new_w = h, w
                # 创建长宽均为原图像2倍的空白图片，将原图像放置在正中央
                canvas_h, canvas_w = new_h * 2, new_w * 2
                final_image = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                final_image.fill(255)  # 填充白色背景
                
                # 计算居中位置
                start_y = (canvas_h - new_h) // 2
                start_x = (canvas_w - new_w) // 2
                
                # 确保不会超出边界
                end_y = min(start_y + new_h, canvas_h)
                end_x = min(start_x + new_w, canvas_w)
                
                # 将原图像放置在中央，确保只复制有效的区域
                src_h = end_y - start_y
                src_w = end_x - start_x
                
                if src_h > 0 and src_w > 0:
                    final_image[start_y:end_y, start_x:end_x] = cropped_image[0:src_h, 0:src_w]
                
                processed_image = final_image
                print(f"图像已放置在{canvas_w}x{canvas_h}画布中央: 位置 ({start_x}, {start_y})")
            
            # 使用临时文件来确保文件名唯一且安全
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            cv2.imwrite(temp_path, processed_image)
            print(f"已保存临时图像: {temp_path}")
            result = self.recognize_text_in_image(temp_path)
            print(f"已完成图像处理: {image_name}")
            result['bbox_coords'] = bbox_coords
            result['image_name'] = image_name
            
            # 在终端输出识别结果
            if result['success']:
                data = result['data']
                extracted_content = self._extract_content_from_result(data)
                result['extracted_content'] = extracted_content
                print(f"  - 识别结果: {extracted_content}")
                
                # 详细输出，用于调试
                print(f"  - 原始数据类型: {type(data)}")
                if isinstance(data, dict):
                    print(f"  - 字典键: {data.keys()}")
                    if 'parsing_res_list' in data:
                        print(f"  - parsing_res_list: {data['parsing_res_list']}")
            else:
                print(f"  - 识别失败: {result.get('error', 'Unknown error')}")
            
            return result
        except Exception as e:
            print(f"处理图像 {image_name} 时出错: {e}")
            # 返回错误结果而不是抛出异常，以确保程序继续执行
            return {
                'success': False,
                'error': str(e),
                'bbox_coords': bbox_coords,
                'image_name': image_name,
                'image_path': temp_path
            }
        finally:
            # 删除临时文件
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if temp_path:
                print(f"已清理临时文件: {temp_path}")

    def _extract_content_from_result(self, data):
        """
        从PaddleOCR结果中提取content字段内容
        """
        import re
        
        # 检查是否是PaddleOCRVLResult对象（根据终端输出，这是一个特殊对象）
        if hasattr(data, 'parsing_res_list'):
            # 这是PaddleOCRVLResult对象，直接获取其parsing_res_list属性
            parsing_list = data.parsing_res_list
        elif isinstance(data, dict) and "parsing_res_list" in data:
            # 普通字典格式
            parsing_list = data.get("parsing_res_list")
        else:
            # 如果不是预期格式，转换为字符串后处理
            text_str = str(data)
            # 尝试从字符串表示中提取content与#之间的内容
            content_match = re.search(r'content:\s*(.+?)(?=\n|\r\n|#################)', text_str, re.DOTALL)
            if content_match:
                return content_match.group(1).strip()
            else:
                # 尝试从多行文本中找到content行
                lines = text_str.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.lower().startswith('content:'):
                        return line[8:].strip()  # 去掉'content:'前缀
            # 如果都没找到，返回空字符串
            return ""
        print(f"########{str(parsing_list)[30:]}")
        # 直接将parsing_list转换为str进行正则
        parsing_str = str(parsing_list)
        # 使用正则提取content与#之间的内容
        content_match = re.search(r'content:\s*(.+?)(?=\n|\r\n|#################)', parsing_str, re.DOTALL)
        if content_match:
            return content_match.group(1).strip()
        else:
            # 尝试从多行文本中找到content行
            lines = parsing_str.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith('content:'):
                    return line[8:].strip()  # 去掉'content:'前缀
        # 如果都没找到，返回空字符串
        return ""


    def recognize_texts_concurrent(self, cropped_images_data: List[Dict[str, Any]], 
                                  max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        顺序识别多个裁剪图像中的文本（由于PaddleOCR线程安全问题，并发处理可能导致错误）
        
        Args:
            cropped_images_data: 包含裁剪图像和相关信息的列表
            max_workers (int): 最大并发数（保留参数兼容性，但实际使用顺序处理）
            
        Returns:
            List[Dict[str, Any]]: 识别结果列表
        """
        results = []
        
        print(f"开始顺序处理 {len(cropped_images_data)} 个图像（由于PaddleOCR线程安全问题，并发处理可能导致错误）")
        
        # 由于PaddleOCR模型存在线程安全问题，使用顺序处理
        # 验证结果显示并发处理会导致PreconditionNotMetError错误
        completed_count = 0
        for data in cropped_images_data:
            try:
                print(f"顺序处理第 {completed_count + 1}/{len(cropped_images_data)} 个图像")
                result = self.recognize_text_in_cropped_image(data['image'], data['name'], data['bbox'])
                results.append(result)
                completed_count += 1
            except Exception as e:
                print(f"顺序处理文本识别时出错: {e}")
                # 记录错误而不是抛出异常，以确保程序继续执行
                results.append({
                    'success': False,
                    'error': str(e),
                    'image_name': data['name'],
                    'bbox_coords': data['bbox']
                })
                completed_count += 1  # 增加计数以确保进度显示正确
        
        print(f"文本识别处理完成，共处理 {len(results)} 个结果")
        return results