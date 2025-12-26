# -*- coding: utf-8 -*-
"""
GLiClass模型推理专用代码
"""
import torch
from modelscope.models import AutoModelForSequenceClassification
from modelscope.tokenizers import AutoTokenizer

# 配置
MODEL_PATH = "./gliclass_circuit_model/best_model"  # 训练好的模型路径
MAX_LENGTH = 20  # 和训练时保持一致

class CircuitTextClassifier:
    def __init__(self):
        # 加载模型和Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict(self, text):
        """单条文本预测"""
        # 预处理
        text = text.lower().strip()
        # 编码
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 解析结果
        pred_id = torch.argmax(outputs.logits, dim=1).item()
        pred_label = self.id2label[pred_id]
        pred_conf = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()
        return {
            "原始文本": text,
            "预测类别": pred_label,
            "置信度": round(pred_conf, 4)
        }

# 测试
if __name__ == "__main__":
    classifier = CircuitTextClassifier()
    
    # 自定义测试文本
    test_texts = [
        "R10 2.2kΩ", "C15 47nF", "STM32F407", 
        "BT-MODE\\SPI_CS_B/WAKE_UP", "top view", "GND",
        "RV1 470V", "D2 0.5A 200V"
    ]
    
    print("===== 电路图文本分类结果 =====")
    for text in test_texts:
        res = classifier.predict(text)
        print(f"{res['原始文本']:25} → {res['预测类别']:8}（置信度：{res['置信度']}）")