# -*- coding: utf-8 -*-
"""
GLiClass模型训练+推理完整代码（适配GLiClass专用模型）
核心修复：
1. 使用GLiClass专用模型类替代AutoModelForSequenceClassification
2. 使用transformers.Trainer替代ModelScope的EpochBasedTrainer
3. 修复accelerate库版本兼容性问题
4. 保留全部功能：日志、FP16、梯度累积、保存3个模型、详细评估、完整推理
环境：Python3.9 + transformers==4.36.0 + accelerate==0.25.0 + torch2.1.0 + gliclass
"""

# ====================== 0. 修复OMP错误：必须在所有import之前设置 ======================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决libiomp5md.dll冲突问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer警告
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"  # 关闭transformers警告

import torch
import pandas as pd
import numpy as np
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== 1. 导入模块（适配GLiClass训练） ======================
# 核心库
from transformers import (
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import Dataset
import evaluate

# GLiClass专用模块
from gliclass import GLiClassModel
from modelscope import AutoTokenizer

# ====================== 2. 全功能配置（无任何删减） ======================
MODEL_ID = "knowledgator/gliclass-large-v3.0"
DATA_DIR = "./text_part/data/dataset"
# 训练参数（全量保留你的配置）
MAX_LENGTH = 20
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 3e-5
# 保存/日志配置（保留所有日志功能）
OUTPUT_DIR = "./text_part/gliclass_circuit_model"
LOG_DIR = "./text_part/gliclass_logs"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ====================== 3. 纯NumPy评估指标（全功能保留） ======================
def calculate_metrics(eval_pred):
    """适配Trainer的评估函数（准确率+加权F1）"""
    logits, labels = eval_pred
    # 处理可能的元组输出
    if isinstance(logits, tuple):
        logits = logits[0]
    
    preds = np.argmax(logits, axis=1)
    
    # 准确率
    accuracy = np.mean(preds == labels)
    
    # 加权F1-score
    unique_labels = np.unique(labels)
    f1_scores = []
    weights = []
    
    for label in unique_labels:
        tp = np.sum((preds == label) & (labels == label))
        fp = np.sum((preds == label) & (labels != label))
        fn = np.sum((preds != label) & (labels == label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        weight = np.sum(labels == label) / len(labels)
        
        f1_scores.append(f1)
        weights.append(weight)
    
    weighted_f1 = np.sum(np.array(f1_scores) * np.array(weights))
    
    return {"accuracy": accuracy, "f1": weighted_f1}

def print_classification_report(preds, labels, label2id):
    """全功能分类报告（纯NumPy实现）"""
    id2label = {v: k for k, v in label2id.items()}
    unique_labels = sorted(np.unique(labels))
    
    print("\n===== 每类分类详细报告 =====")
    print(f"{'类别':<10} {'精准率':<10} {'召回率':<10} {'F1-score':<10} {'样本数':<10}")
    print("-" * 50)
    
    for label in unique_labels:
        tp = np.sum((preds == label) & (labels == label))
        fp = np.sum((preds == label) & (labels != label))
        fn = np.sum((preds != label) & (labels == label))
        support = np.sum(labels == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{id2label[label]:<10} {precision:.4f}      {recall:.4f}      {f1:.4f}      {support:<10}")
    
    metrics = calculate_metrics((preds, labels))
    print("-" * 50)
    print(f"{'加权平均':<10} {'':<10} {'':<10} {metrics['f1']:.4f}      {len(labels):<10}")
    print(f"{'准确率':<10} {'':<10} {'':<10} {metrics['accuracy']:.4f}      {len(labels):<10}")

# ====================== 4. 数据加载（全功能保留） ======================
def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在：{csv_path}")
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    required_cols = ["text", "label"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据文件缺少必要列：{required_cols}")
    
    label_list = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    if "label_id" not in df.columns:
        df["label_id"] = df["label"].map(label2id)
    
    id2label = {idx: label for label, idx in label2id.items()}
    return df, label2id, id2label

# 加载数据（保留原逻辑）
train_df, label2id, id2label = load_data(os.path.join(DATA_DIR, "train_data.csv"))
val_df, _, _ = load_data(os.path.join(DATA_DIR, "val_data.csv"))
test_df, _, _ = load_data(os.path.join(DATA_DIR, "test_data.csv"))
num_labels = len(label2id)
print(f"✅ 数据加载完成：训练集{len(train_df)}条，验证集{len(val_df)}条，测试集{len(test_df)}条")

# ====================== 5. GLiClass专用：Tokenizer+模型加载 ======================
print("===== 加载GLiClass模型和Tokenizer（GLiClass专用） =====")
# 加载Tokenizer（保持非fast版本解决兼容性问题）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False
)
# 保留特殊符号扩展
special_tokens = ["Ω", "μF", "nF", "pF", "mH", "μH", "\\", "/", "-", "_"]
tokenizer.add_tokens(special_tokens)
print(f"✅ Tokenizer扩展完成，新增{len(special_tokens)}个特殊符号")

# 关键修改：使用GLiClass专用模型类
print("===== 加载GLiClass专用模型 =====")
model = GLiClassModel.from_pretrained(
    MODEL_ID,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    trust_remote_code=True
)

# 保留嵌入层更新
model.resize_token_embeddings(len(tokenizer))
print("✅ GLiClass模型加载完成，嵌入层已更新")

# ====================== 6. 数据集构建（适配transformers.Trainer） ======================
def tokenize_function(examples):
    """适配transformers.Trainer的tokenization函数"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# 转换为Dataset格式
train_dataset = Dataset.from_dict({
    "text": train_df["text"].tolist(),
    "label": train_df["label_id"].tolist()
})
val_dataset = Dataset.from_dict({
    "text": val_df["text"].tolist(),
    "label": val_df["label_id"].tolist()
})
test_dataset = Dataset.from_dict({
    "text": test_df["text"].tolist(),
    "label": test_df["label_id"].tolist()
})

# 应用tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置格式
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print(f"✅ 数据集格式转换完成：训练集{len(train_dataset)}条，验证集{len(val_dataset)}条，测试集{len(test_dataset)}条")

# ====================== 7. 训练参数配置（全功能保留，无任何删减） ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,          # 保留独立日志目录
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_strategy="epoch",        # 修复：evaluation_strategy → eval_strategy
    save_strategy="epoch",        # 保留每轮保存
    load_best_model_at_end=True,  # 保留加载最优模型
    metric_for_best_model="f1",
    logging_steps=10,             # 保留每10步日志
    fp16=torch.cuda.is_available(),  # 保留GPU FP16
    gradient_accumulation_steps=2,   # 保留梯度累积
    save_total_limit=3,              # 保留最多保存3个模型
    remove_unused_columns=False,
    seed=42,                         # 保留固定种子
    report_to="tensorboard",         # 保留日志功能
    disable_tqdm=False,              # 保留进度条
    run_name="gliclass_circuit_training",
    logging_first_step=True,         # 记录第一步
    eval_accumulation_steps=10,      # 评估时累积步骤
    dataloader_num_workers=0,        # Windows上设置为0避免多进程问题
    dataloader_pin_memory=True       # 启用pin_memory加速
)

# ====================== 8. 模型训练（使用transformers.Trainer） ======================
print("===== 开始训练GLiClass模型 =====")

class GLiClassTrainer(Trainer):
    """自定义Trainer以适配GLiClassModel"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """自定义损失计算（带详细调试信息）"""
        # 处理标签字段
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.pop("label")
        else:
            inputs.pop("labels")
        
        # 调试：打印输入形状
        batch_size = labels.shape[0]
        print(f"\n🔍 输入批次大小: {batch_size}")
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key} 形状: {value.shape}")
        
        # 模型前向传播
        outputs = model(**inputs)
        
        # 调试：分析模型输出
        print(f"🔍 模型输出类型: {type(outputs)}")
        
        # 获取logits（处理不同输出格式）
        logits = None
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            print(f"  ✓ 使用 outputs.logits，形状: {logits.shape}")
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            print(f"  ⚠️  输出是元组，长度: {len(outputs)}")
            for i, item in enumerate(outputs):
                if hasattr(item, 'shape'):
                    print(f"    元素 {i} 形状: {item.shape}")
                    # 寻找形状匹配的元素
                    if len(item.shape) >= 2 and item.shape[0] == batch_size:
                        logits = item
                        print(f"    ✓ 选择元素 {i} 作为logits")
                        break
            if logits is None and len(outputs) > 0:
                logits = outputs[0]
                print(f"    ⚠️  使用第一个元素作为logits，形状: {logits.shape}")
        elif hasattr(outputs, 'shape'):
            logits = outputs
            print(f"  ✓ 使用直接输出，形状: {logits.shape}")
        else:
            raise ValueError(f"无法从模型输出中获取logits，输出类型: {type(outputs)}")
        
        if logits is None:
            raise ValueError("无法获取有效的logits")
        
        # 调试：打印logits和labels形状
        print(f"🔍 Logits 形状: {logits.shape}, Labels 形状: {labels.shape}")
        
        # 确保logits是2D或3D张量
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
            print(f"  ⚠️  将1D logits扩展为2D，新形状: {logits.shape}")
        
        # 检查形状是否匹配
        if logits.shape[0] != batch_size:
            print(f"  ⚠️  Logits批次大小 ({logits.shape[0]}) 与标签批次大小 ({batch_size}) 不匹配")
            # 尝试调整形状
            if logits.shape[0] == 1 and logits.shape[1] == batch_size * self.model.config.num_labels:
                logits = logits.view(batch_size, self.model.config.num_labels)
                print(f"    ✓ 重新调整形状，新形状: {logits.shape}")
            elif logits.shape[1] == batch_size and logits.shape[2] == self.model.config.num_labels:
                logits = logits[0]  # 取第一个维度
                print(f"    ✓  取第一个维度，新形状: {logits.shape}")
        
        # 再次检查
        print(f"🔍 调整后 Logits 形状: {logits.shape}, Labels 形状: {labels.shape}")
        
        if logits.shape[0] != labels.shape[0]:
            raise ValueError(f"Logits批次大小 ({logits.shape[0]}) 与标签批次大小 ({labels.shape[0]}) 仍然不匹配")
        
        # 计算损失
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        print(f"  ✓ 损失计算成功: {loss.item():.4f}")
        
        return (loss, logits) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """自定义预测步骤"""
        # 调用父类方法
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys
        )
        
        # 确保logits是numpy数组
        if logits is not None and not isinstance(logits, np.ndarray):
            logits = logits.detach().cpu().numpy()
        
        # 确保labels是numpy数组
        if labels is not None and not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()
        
        return loss, logits, labels

trainer = GLiClassTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda p: calculate_metrics((p.predictions, p.label_ids)),
    tokenizer=tokenizer
)

# 启动训练（保留原逻辑）
print("🚀 开始训练过程...")
train_result = trainer.train()

# 记录训练结果
print("\n===== 训练完成 =====")
print(f"总训练步数: {train_result.global_step}")
print(f"训练损失: {train_result.training_loss:.4f}")

# ====================== 9. 模型评估（全功能保留） ======================
print("\n===== 测试集评估 =====")
test_results = trainer.evaluate(test_dataset)
print(f"测试集加权F1-score: {test_results['eval_f1']:.4f}")
print(f"测试集准确率: {test_results['eval_accuracy']:.4f}")
print(f"测试集损失: {test_results['eval_loss']:.4f}")

# 生成详细报告
print("\n===== 生成详细分类报告 =====")
test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=-1)

# 确保标签数组正确
if isinstance(test_dataset["label"], list):
    test_labels = np.array(test_dataset["label"])
else:
    test_labels = test_dataset["label"]

print_classification_report(test_preds, test_labels, label2id)

# ====================== 10. 模型保存（全功能保留） ======================
print("\n===== 保存最优GLiClass模型 =====")
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

# 保存训练参数
import json
with open(os.path.join(BEST_MODEL_DIR, "training_args.json"), "w") as f:
    json.dump(training_args.to_dict(), f, indent=2)

print(f"✅ GLiClass模型已保存至：{BEST_MODEL_DIR}")
print(f"✅ 训练参数已保存至：{os.path.join(BEST_MODEL_DIR, 'training_args.json')}")

# ====================== 11. 推理函数（使用GLiClass模型） ======================
class CircuitClassifier:
    """全功能推理类（使用GLiClass专用模型）"""
    def __init__(self, model_path):
        print(f"===== 加载推理模型：{model_path} =====")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        # 使用GLiClass专用模型进行推理
        self.model = GLiClassModel.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = id2label
        print(f"✅ 推理模型加载完成，设备：{self.device}")

    def predict(self, text):
        """保留原单条推理逻辑（适配GLiClass）"""
        text = text.lower().strip()
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # GLiClass模型输出处理
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            logits = outputs[0]
        else:
            logits = outputs
        
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = self.id2label[pred_id]
        pred_conf = torch.softmax(logits, dim=1)[0][pred_id].item()
        
        return {
            "text": text,
            "label": pred_label,
            "confidence": round(pred_conf, 4)
        }

    def batch_predict(self, texts):
        """保留批量推理"""
        results = []
        for i, text in enumerate(texts):
            result = self.predict(text)
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"已处理 {i+1}/{len(texts)} 条文本")
        return results

# ====================== 12. 推理示例（全功能保留） ======================
if __name__ == "__main__":
    print("\n\n" + "="*50)
    print("===== 启动推理示例 =====")
    classifier = CircuitClassifier(BEST_MODEL_DIR)
    
    # 保留原测试样本
    test_samples = [
        "R1 100Ω", "C2 10μF", "L3 22mH", "U4 STM32F103",
        "M5 12V", "D6 1A 400V", "GND7", "RV8 220V"
    ]
    
    print("\n===== GLiClass推理示例 =====")
    results = classifier.batch_predict(test_samples)
    for res in results:
        print(f"文本：{res['text']:15} → 类别：{res['label']:8} → 置信度：{res['confidence']:.4f}")
    
    print("\n✅ 所有流程执行完成！")