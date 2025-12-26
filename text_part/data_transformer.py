# data_transformer.py 完整修复版
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ====================== 1. 加载并清洗数据 ======================
# 加载CSV（确保circuit_data.csv和此脚本同目录，或填写绝对路径）
df = pd.read_csv("./text_part/data/circuit_data.csv", encoding="utf-8")

# 数据清洗（保留特殊符号，统一大小写、去冗余空格）
df["text"] = df["text"].str.lower().str.strip()  # 统一小写、去空格
df = df.dropna(subset=["text", "label"])  # 删除空值
df = df.reset_index(drop=True)  # 重置索引

# ====================== 2. 标签编码（类别→数字） ======================
label_list = sorted(df["label"].unique())
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# 打印数据基本信息
print(f"类别列表（共{len(label_list)}类）：{label_list}")
print(f"清洗后数据总量：{len(df)}条")
print(f"各类型数量：\n{df['label'].value_counts()}")

# ====================== 3. 划分训练/验证/测试集（8:1:1） ======================
# 分层划分，保证每类数据在各集合中分布均匀
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

# ====================== 4. 保存划分后的数据（可选，方便后续查看） ======================
train_df.to_csv("./text_part/data/dataset/train_data.csv", index=False, encoding="utf-8")
val_df.to_csv("./text_part/data/dataset/val_data.csv", index=False, encoding="utf-8")
test_df.to_csv("./text_part/data/dataset/test_data.csv", index=False, encoding="utf-8")

# ====================== 5. 构建模型输入的数据集格式（列表+字典，替代MsDataset） ======================
# 无需ModelScope Dataset，直接构建原生Python数据集（后续模型训练可直接用）
def build_dataset(dataframe):
    """将DataFrame转为模型可直接使用的列表格式"""
    texts = dataframe["text"].tolist()
    labels = dataframe["label_id"].tolist()
    return [{"text": t, "label": l} for t, l in zip(texts, labels)]

# 构建训练/验证/测试集
train_dataset = build_dataset(train_df)
val_dataset = build_dataset(val_df)
test_dataset = build_dataset(test_df)

# 打印划分结果
print("\n数据集划分完成：")
print(f"训练集：{len(train_dataset)}条，验证集：{len(val_dataset)}条，测试集：{len(test_dataset)}条")

# ====================== 可选：验证数据格式 ======================
print("\n数据格式示例（前3条训练数据）：")
for i in range(3):
    print(f"文本：{train_dataset[i]['text']} → 标签ID：{train_dataset[i]['label']} → 标签名：{id2label[train_dataset[i]['label']]}")