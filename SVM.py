# -*- coding: utf-8 -*-
"""
超短电子元件文本分类（纯Python无编译依赖版）
仅依赖pandas + scikit-learn，无需fasttext/大模型
"""
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ====================== 1. 核心特征提取函数（适配电子元件文本） ======================
def extract_component_features(text):
    """
    提取超短电子元件文本的结构化特征（核心！）
    特征包括：前缀、单位、关键词、文本长度等
    """
    text = text.lower().strip()
    features = {}
    
    # 特征1：元件前缀（最核心的分类依据）
    # 匹配开头的字母/字母组合（如R/C/L/U/M/D/GND/RV）
    prefix_pattern = r'^([a-z]+)'
    prefix_match = re.search(prefix_pattern, text)
    if prefix_match:
        prefix = prefix_match.group(1)
        features['prefix'] = prefix  # 如r/c/l/u/m/d/gnd/rv等
    
    # 特征2：单位特征（辅助分类）
    unit_to_type = {
        'ω': '电阻', 'ohm': '电阻', '欧姆': '电阻',
        'uf': '电容', 'μf': '电容', 'nf': '电容', 'pf': '电容', 'mf': '电容',
        'mh': '电感', 'μh': '电感', 'nh': '电感',
        'v': '电压', 'a': '电流'
    }
    # 检查文本中是否包含目标单位
    for unit, unit_type in unit_to_type.items():
        if unit in text:
            features['unit_type'] = unit_type
            break
    
    # 特征3：关键词特征（针对接地/芯片等特殊类别）
    if 'gnd' in text:
        features['keyword'] = 'gnd'  # 接地
    elif 'stm' in text or 'ic' in text or 'chip' in text:
        features['keyword'] = 'chip'  # 芯片
    elif 'motor' in text or 'motor' in text:
        features['keyword'] = 'motor'  # 电机
    elif 'diode' in text:
        features['keyword'] = 'diode'  # 二极管
    elif 'varistor' in text or 'rv' == features.get('prefix', ''):
        features['keyword'] = 'varistor'  # 压敏电阻
    
    # 特征4：文本长度（辅助区分异常文本）
    features['text_length'] = len(text)
    
    return features

# ====================== 2. 加载数据 ======================
# 替换为你的数据路径（和之前GLiClass代码路径一致）
TRAIN_PATH = "./text_part/data/dataset/train_data.csv"
VAL_PATH = "./text_part/data/dataset/val_data.csv"
TEST_PATH = "./text_part/data/dataset/test_data.csv"

# 加载并合并数据
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='utf-8')
    # 确保必要列存在
    assert "text" in df.columns and "label" in df.columns, "数据缺少text/label列"
    return df

df_train = load_data(TRAIN_PATH)
df_val = load_data(VAL_PATH)
df_test = load_data(TEST_PATH)
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

# 标签映射（和你之前的标签一致）
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

# 提取特征 + 标签编码
df_all['features'] = df_all['text'].apply(extract_component_features)
df_all['label_id'] = df_all['label'].map(label2id)

# ====================== 3. 特征向量化 ======================
# 将字典格式的特征转为数值特征（SVM需要数值输入）
vec = DictVectorizer(sparse=False)
X = vec.fit_transform(df_all['features'].tolist())  # 特征矩阵
y = df_all['label_id'].values                       # 标签数组

# 拆分训练集/测试集（按8:2拆分，保证标签分布一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # 分层采样，避免标签不平衡
)

# ====================== 4. 训练SVM分类器（核心） ======================
# SVM特别适合高维稀疏特征，对超短文本的结构化特征效果极佳
clf = SVC(
    kernel='linear',        # 线性核（适配小数据、高维特征）
    C=1.0,                  # 正则化强度
    class_weight='balanced',# 平衡类别权重（解决样本不平衡）
    random_state=42,        # 固定随机种子
    probability=True        # 开启概率预测（方便输出置信度）
)
clf.fit(X_train, y_train)

# ====================== 5. 模型评估 ======================
# 测试集预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ 测试集准确率：{accuracy:.4f}")

# 详细分类报告（看每类的精准率/召回率）
print("\n📊 每类分类详细报告：")
print(classification_report(
    [id2label[i] for i in y_test],  # 真实标签
    [id2label[i] for i in y_pred],  # 预测标签
    target_names=label2id.keys()
))

# ====================== 6. 推理函数（直接用！） ======================
def predict_component(text):
    """
    单条文本推理函数
    参数：text - 电子元件文本（如"R1 100Ω"）
    返回：字典，包含文本、预测类别、置信度
    """
    # 提取特征
    features = extract_component_features(text)
    # 特征向量化（用训练好的Vectorizer）
    X = vec.transform([features])
    # 预测
    pred_id = clf.predict(X)[0]
    pred_label = id2label[pred_id]
    # 计算置信度
    pred_prob = clf.predict_proba(X)[0][pred_id]
    
    return {
        "text": text,
        "label": pred_label,
        "confidence": round(pred_prob, 4)
    }

# ====================== 7. 测试推理 ======================
if __name__ == "__main__":
    print("\n🔍 推理测试示例：")
    test_samples = [
        "R1 100Ω", "C2 10μF", "L3 22mH", "U4 STM32F103",
        "M5 12V", "D6 1A 400V", "GND7", "RV8 220V"
    ]
    
    for sample in test_samples:
        result = predict_component(sample)
        print(f"文本：{result['text']:15} → 类别：{result['label']:8} → 置信度：{result['confidence']}")

# ====================== 8. 保存模型（可选，方便后续复用） ======================
import joblib
# 保存分类器和特征转换器
joblib.dump(clf, "./text_part/component_classifier.pkl")
joblib.dump(vec, "./text_part/feature_vectorizer.pkl")
print("\n💾 模型已保存：./text_part/component_classifier.pkl + ./text_part/feature_vectorizer.pkl")

# 加载模型的方法（后续推理用）
# clf = joblib.load("./component_classifier.pkl")
# vec = joblib.load("./feature_vectorizer.pkl")