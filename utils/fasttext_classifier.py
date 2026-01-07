import fasttext
import os
from typing import Dict, Any


class FastTextComponentClassifier:
    """
    使用FastText模型对电路元件文本进行分类的类
    """
    
    def __init__(self, model_path: str = "component_fasttext.bin"):
        """
        初始化FastText分类器
        
        Args:
            model_path: FastText模型文件路径
        """
        self.model_path = model_path
        self.model = None
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"FastText模型文件不存在: {self.model_path}")
        
        # 加载模型
        self.model = fasttext.load_model(self.model_path)
        
    def predict(self, text: str) -> Dict[str, Any]:
        """
        对输入文本进行分类预测
        
        Args:
            text: 待分类的文本
            
        Returns:
            包含文本、预测标签和置信度的字典
        """
        if not self.model:
            raise RuntimeError("FastText模型未加载")
            
        # 预处理文本
        processed_text = text.lower().strip()
        
        try:
            # 进行预测
            pred = self.model.predict(processed_text, k=1)
            label = pred[0][0].replace("__label__", "")
            confidence = pred[1][0]
            
            return {
                'text': text,
                'predicted_label': label,
                'confidence': round(confidence, 4)
            }
        except Exception as e:
            # 如果预测失败，返回默认值
            print(f"FastText预测失败: {e}")
            return {
                'text': text,
                'predicted_label': 'unknown',
                'confidence': 0.0
            }
    
    def predict_batch(self, texts: list) -> list:
        """
        批量对文本进行分类预测
        
        Args:
            texts: 待分类的文本列表
            
        Returns:
            预测结果列表
        """
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                print(f"FastText批量预测中处理文本 '{text}' 时失败: {e}")
                # 添加默认结果
                results.append({
                    'text': text,
                    'predicted_label': 'unknown',
                    'confidence': 0.0
                })
        return results