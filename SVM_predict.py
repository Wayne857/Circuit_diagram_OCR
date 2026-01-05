# -*- coding: utf-8 -*-
"""
电子元件文本分类 - 模型加载与预测脚本
依赖：pandas、scikit-learn、joblib
"""
import os
import re
import joblib
import pandas as pd
import numpy as np

# ====================== 1. 配置项（根据实际保存路径修改） ======================
# 模型/特征转换器保存路径
CLASSIFIER_PATH = "./text_part/component_classifier.pkl"
VECTORIZER_PATH = "./text_part/feature_vectorizer.pkl"
# 标签映射（和训练时一致）
label2id = {
    '二极管': 0,
    '压敏电阻': 1,
    '接地': 2,
    '电容': 3,
    '电感': 4,
    '电机': 5,
    '电阻': 6,
    '芯片': 7
}
id2label = {v: k for k, v in label2id.items()}

# ====================== 2. 核心特征提取函数（和训练时完全一致！） ======================
def extract_component_features(text):
    """
    提取超短电子元件文本的结构化特征
    注意：必须和训练时的特征提取逻辑完全一致，否则预测会出错
    """
    text = text.lower().strip()
    features = {}
    
    # 特征1：元件前缀
    prefix_pattern = r'^([a-z]+)'
    prefix_match = re.search(prefix_pattern, text)
    if prefix_match:
        prefix = prefix_match.group(1)
        features['prefix'] = prefix
    
    # 特征2：单位特征
    unit_to_type = {
        'ω': '电阻', 'ohm': '电阻', '欧姆': '电阻',
        'uf': '电容', 'μf': '电容', 'nf': '电容', 'pf': '电容', 'mf': '电容',
        'mh': '电感', 'μh': '电感', 'nh': '电感',
        'v': '电压', 'a': '电流'
    }
    for unit, unit_type in unit_to_type.items():
        if unit in text:
            features['unit_type'] = unit_type
            break
    
    # 特征3：关键词特征
    if 'gnd' in text:
        features['keyword'] = 'gnd'
    elif 'stm' in text or 'ic' in text or 'chip' in text:
        features['keyword'] = 'chip'
    elif 'motor' in text or 'motor' in text:
        features['keyword'] = 'motor'
    elif 'diode' in text:
        features['keyword'] = 'diode'
    elif 'varistor' in text or 'rv' == features.get('prefix', ''):
        features['keyword'] = 'varistor'
    
    # 特征4：文本长度
    features['text_length'] = len(text)
    
    return features

# ====================== 3. 模型加载类（封装预测逻辑） ======================
class ComponentClassifier:
    def __init__(self):
        """初始化：加载模型和特征转换器"""
        self._check_model_files()
        print("🔍 正在加载模型...")
        
        # 加载分类器和特征转换器
        self.clf = joblib.load(CLASSIFIER_PATH)
        self.vec = joblib.load(VECTORIZER_PATH)
        print(f"✅ 模型加载完成：")
        print(f"   - 分类器：{CLASSIFIER_PATH}")
        print(f"   - 特征转换器：{VECTORIZER_PATH}")
    
    def _check_model_files(self):
        """检查模型文件是否存在"""
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(f"分类器文件不存在：{CLASSIFIER_PATH}")
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"特征转换器文件不存在：{VECTORIZER_PATH}")
    
    def predict_single(self, text):
        """
        单条文本预测
        :param text: 电子元件文本（如"R1 100Ω"）
        :return: dict - 包含文本、预测类别、置信度
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {"text": text, "label": "无效文本", "confidence": 0.0}
        
        # 提取特征 + 向量化
        features = extract_component_features(text)
        X = self.vec.transform([features])
        
        # 预测
        pred_id = self.clf.predict(X)[0]
        pred_label = id2label[pred_id]
        pred_prob = self.clf.predict_proba(X)[0][pred_id]
        
        return {
            "text": text.strip(),
            "label": pred_label,
            "confidence": round(pred_prob, 4)
        }
    
    def predict_batch(self, texts, save_to_csv=False, csv_path="./prediction_result.csv"):
        """
        批量文本预测
        :param texts: 文本列表（如["R1 100Ω", "C2 10μF"]）
        :param save_to_csv: 是否保存结果到CSV
        :param csv_path: CSV保存路径
        :return: list - 包含所有预测结果的字典列表
        """
        if not isinstance(texts, list) or len(texts) == 0:
            print("⚠️  批量预测输入为空！")
            return []
        
        print(f"\n📦 开始批量预测（共{len(texts)}条）...")
        results = []
        
        # 批量提取特征 + 向量化
        features_list = [extract_component_features(text) for text in texts]
        X = self.vec.transform(features_list)
        
        # 批量预测
        pred_ids = self.clf.predict(X)
        pred_probs = self.clf.predict_proba(X)
        
        # 整理结果
        for idx, text in enumerate(texts):
            pred_id = pred_ids[idx]
            pred_label = id2label[pred_id]
            confidence = round(pred_probs[idx][pred_id], 4)
            
            result = {
                "text": text.strip(),
                "label": pred_label,
                "confidence": confidence
            }
            results.append(result)
        
        # 保存到CSV
        if save_to_csv:
            df_result = pd.DataFrame(results)
            df_result.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"✅ 批量预测结果已保存至：{csv_path}")
        
        return results

# ====================== 4. 预测示例（直接运行即可） ======================
if __name__ == "__main__":
    # 初始化分类器
    classifier = ComponentClassifier()
    
    # -------------------- 示例1：单条预测 --------------------
    print("\n" + "="*50)
    print("📝 单条预测示例")
    print("-"*50)
    test_samples = [
        "R1 100Ω", "C2 10μF", "L3 22mH", "U4 STM32F103",
        "M5 12V", "D6 1A 400V", "GND7", "RV8 220V",
        "X9 10V",  # 边缘案例
        "",        # 空文本
        "C4",
        "LDM",
        "CX",
        "AD-DC",
        "MOV",
        "BT_MODE",
        "SDO",
        "TXD",
        "VDD",
        "无效文本" # 无效文本
    ]
    
    for sample in test_samples:
        res = classifier.predict_single(sample)
        print(f"文本：{res['text']:15} → 类别：{res['label']:8} → 置信度：{res['confidence']}")
    
    # -------------------- 示例2：批量预测 --------------------
    # print("\n" + "="*50)
    # print("📝 批量预测示例（保存到CSV）")
    # print("-"*50)
    # batch_texts = [
    #     "R10 200Ω", "C11 22μF", "L12 10mH", "U13 ATmega328",
    #     "M14 24V", "D15 2A 600V", "GND16", "RV17 110V"
    # ]
    
    # # 批量预测并保存到CSV
    # batch_results = classifier.predict_batch(batch_texts, save_to_csv=True)
    
    # # 打印批量结果
    # for res in batch_results:
    #     print(f"文本：{res['text']:15} → 类别：{res['label']:8} → 置信度：{res['confidence']}")
    
    # # -------------------- 示例3：从CSV文件读取文本批量预测 --------------------
    # print("\n" + "="*50)
    # print("📝 从CSV文件读取文本批量预测")
    # print("-"*50)
    # # 假设你有一个待预测的CSV文件，包含"text"列
    # # 先创建测试文件（实际使用时替换为你的文件路径）
    # test_csv_path = "./to_predict.csv"
    # pd.DataFrame({"text": batch_texts}).to_csv(test_csv_path, index=False, encoding="utf-8")
    
    # # 读取CSV并预测
    # df_to_predict = pd.read_csv(test_csv_path, encoding="utf-8")
    # texts_from_csv = df_to_predict["text"].tolist()
    # csv_results = classifier.predict_batch(texts_from_csv, save_to_csv=True, csv_path="./csv_prediction_result.csv")
    
    # print(f"✅ 从CSV预测完成，结果保存至：./text_part/csv_prediction_result.csv")

    print("\n" + "="*50)
    print("🎉 所有预测完成！")