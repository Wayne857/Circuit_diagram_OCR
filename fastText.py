import fasttext
import pandas as pd
import os

# ====================== 1. 准备FastText格式数据 ======================
# FastText要求格式：__label__类别 文本
df = pd.read_csv("./text_part/data/dataset/train_data.csv", encoding='utf-8')
df_val = pd.read_csv("./text_part/data/dataset/val_data.csv", encoding='utf-8')
df_test = pd.read_csv("./text_part/data/dataset/test_data.csv", encoding='utf-8')

# 训练集
with open("fasttext_train.txt", "w", encoding='utf-8') as f:
    for _, row in df.iterrows():
        f.write(f"__label__{row['label']} {row['text'].lower()}\n")

# 验证集
with open("fasttext_val.txt", "w", encoding='utf-8') as f:
    for _, row in df_val.iterrows():
        f.write(f"__label__{row['label']} {row['text'].lower()}\n")

# ====================== 2. 训练FastText ======================
# 核心参数：wordNgrams=2（捕捉双字符组合，如R1、10Ω），epoch=20（小数据足够）
model = fasttext.train_supervised(
    input="fasttext_train.txt",
    lr=0.1,          # 学习率
    dim=32,          # 向量维度（小维度适配短文本）
    ws=5,            # 窗口大小
    epoch=20,
    minCount=1,      # 最小词频
    wordNgrams=2,    # 关键：n-gram=2，捕捉字符组合
    loss='softmax',
    bucket=200000,
    thread=4,
    verbose=2
)

# 验证
val_result = model.test("fasttext_val.txt")
print(f"验证集：样本数={val_result[0]}, 准确率={val_result[1]:.4f}, F1={val_result[2]:.4f}")

# ====================== 3. 推理 ======================
def fasttext_predict(text):
    text = text.lower()
    # k=1：返回最可能的类别
    pred = model.predict(text, k=1)
    label = pred[0][0].replace("__label__", "")
    confidence = pred[1][0]
    return {'text': text, 'label': label, 'confidence': round(confidence, 4)}

# 测试
test_samples = ["R1 100Ω", "C2 10μF", "L3 22mH", "U4 STM32F103", "M5 12V", "D6 1A 400V", "1GND7", "RV8 220V", "1.0μF", "R4 2Ω"]
for sample in test_samples:
    res = fasttext_predict(sample)
    print(f"{res['text']:15} → {res['label']:8} (置信度：{res['confidence']})")

# 保存模型
model.save_model("component_fasttext.bin")