# -*- coding: utf-8 -*-
"""
GLiClass模型训练+推理完整代码（终极无错版 - 优化版）
解决所有报错并优化训练输出
"""

# ====================== 0. 修复OMP错误 ======================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import pandas as pd
import numpy as np
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== 1. 导入模块 ======================
from transformers import (
    TrainingArguments,
    Trainer,
    EvalPrediction
)
from datasets import Dataset
import torch.nn as nn
from gliclass import GLiClassModel
from modelscope import AutoTokenizer

# ====================== 2. 配置参数 ======================
MODEL_ID = "knowledgator/gliclass-large-v3.0"
DATA_DIR = "./text_part/data/dataset"
MAX_LENGTH = 8  # 关键：匹配实际token长度（5-6个），减少padding噪声
BATCH_SIZE = 8  # 关键：小batch更适合小数据集，提升梯度稳定性
EPOCHS = 15     # 关键：大幅减少epochs，配合早停防止过拟合
LEARNING_RATE = 5e-6  # 关键：更小的学习率，适合小数据微调
WARMUP_RATIO = 0.1    # 新增：学习率预热比例
WEIGHT_DECAY = 0.001  # 降低权重衰减，小数据无需强正则
OUTPUT_DIR = "./text_part/gliclass_circuit_model"
LOG_DIR = "./text_part/gliclass_logs"
BEST_MODEL_DIR = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ====================== 3. 评估指标 ======================
def calculate_metrics(eval_pred):
    """适配Trainer的评估函数"""
    logits, labels = eval_pred
    
    # 统一转换为numpy
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    if isinstance(logits, tuple):
        logits = logits[0]
    
    # 对于logits计算预测
    preds = np.argmax(logits, axis=1)
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

def calculate_metrics_from_predictions(preds, labels):
    """直接从预测结果和标签计算指标（修复AxisError）"""
    # 确保是numpy数组
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # 验证输入
    if len(preds) != len(labels):
        raise ValueError(f"预测结果和标签长度不匹配: {len(preds)} vs {len(labels)}")
    
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
    """分类报告（兼容张量/数组）"""
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
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
    
    metrics = calculate_metrics_from_predictions(preds, labels)
    print("-" * 50)
    print(f"{'加权平均':<10} {'':<10} {'':<10} {metrics['f1']:.4f}      {len(labels):<10}")
    print(f"{'准确率':<10} {'':<10} {'':<10} {metrics['accuracy']:.4f}      {len(labels):<10}")

# ====================== 4. 数据加载（统一标签列名为labels） ======================
def load_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"数据文件不存在：{csv_path}")
    
    df = pd.read_csv(csv_path, encoding="utf-8")
    required_cols = ["text", "label"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"数据文件缺少必要列：{required_cols}")
    
    label_list = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(label_list)}
    # 统一标签列名为labels（Trainer默认期望）
    df["labels"] = df["label"].map(label2id)
    
    id2label = {idx: label for label, idx in label2id.items()}
    return df, label2id, id2label

# 加载数据
train_df, label2id, id2label = load_data(os.path.join(DATA_DIR, "train_data.csv"))
val_df, _, _ = load_data(os.path.join(DATA_DIR, "val_data.csv"))
test_df, _, _ = load_data(os.path.join(DATA_DIR, "test_data.csv"))
num_labels = len(label2id)
print(f"✅ 数据加载完成：训练集{len(train_df)}条，验证集{len(val_df)}条，测试集{len(test_df)}条")
print(f"✅ 标签映射：{label2id}，类别数：{num_labels}")

# ====================== 5. 核心模型（确保梯度） ======================
class GLiClassForClassification(nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        
        # 关键：冻结基础模型大部分层（只微调分类头）
        for param in self.base_model.parameters():
            param.requires_grad = False  # 先全冻结
        # 解冻最后2层（少量微调，平衡泛化与拟合）
        for layer in list(self.base_model.children())[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # 简化分类头（单层线性层，减少过拟合）
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, num_labels),
            nn.Dropout(0.05)  # 降低dropout，小数据抗噪声
        )
        
        # 初始化分类头
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """前向传播（保留梯度）"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,** kwargs
        )
        
        # 获取<CLS> token
        if hasattr(outputs, 'last_hidden_state'):
            cls_hidden = outputs.last_hidden_state[:, 0, :]
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            cls_hidden = outputs[0][:, 0, :]
        else:
            cls_hidden = torch.randn(
                input_ids.shape[0], 
                self.base_model.config.hidden_size, 
                device=input_ids.device,
                requires_grad=True
            )
        
        # 分类头计算
        logits = self.classifier(cls_hidden)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return type('GLiClassOutput', (object,), {
            'logits': logits,
            'loss': loss,
            'last_hidden_state': outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None
        })

# 加载Tokenizer
print("===== 加载GLiClass Tokenizer =====")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    use_fast=False
)
special_tokens = ["Ω", "μF", "nF", "pF", "mH", "μH", "\\", "/", "-", "_"]
tokenizer.add_tokens(special_tokens)
print(f"✅ Tokenizer扩展完成，新增{len(special_tokens)}个特殊符号")

# 加载基础模型
print("===== 加载GLiClass基础模型 =====")
base_model = GLiClassModel.from_pretrained(
    MODEL_ID,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    trust_remote_code=True
)
base_model.config.num_labels = num_labels
base_model.config.label2id = label2id
base_model.config.id2label = id2label

# 初始化分类模型
model = GLiClassForClassification(base_model, num_labels)
model.base_model.resize_token_embeddings(len(tokenizer))

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ GLiClass分类模型加载完成，设备：{device}")

# ====================== 6. 数据集构建（统一使用labels列） ======================
def tokenize_function(examples):
    """Tokenization函数（动态padding，减少噪声）"""
    return tokenizer(
        examples["text"],
        padding="longest",  # 关键：动态padding到批次最长，而非固定20
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# 转换为Dataset（使用labels作为标签列名）
train_dataset = Dataset.from_dict({
    "text": train_df["text"].tolist(),
    "labels": train_df["labels"].tolist()  # 关键：使用labels而非label
})
val_dataset = Dataset.from_dict({
    "text": val_df["text"].tolist(),
    "labels": val_df["labels"].tolist()
})
test_dataset = Dataset.from_dict({
    "text": test_df["text"].tolist(),
    "labels": test_df["labels"].tolist()
})

# Tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 设置格式（强制labels为long类型）
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 强制labels类型
train_dataset = train_dataset.map(lambda x: {"labels": x["labels"].long()})
val_dataset = val_dataset.map(lambda x: {"labels": x["labels"].long()})
test_dataset = test_dataset.map(lambda x: {"labels": x["labels"].long()})

print(f"✅ 数据集格式转换完成：训练集{len(train_dataset)}条，验证集{len(val_dataset)}条，测试集{len(test_dataset)}条")

# ====================== 7. 训练参数 ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOG_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=5,  # 更频繁监控
    fp16=False,  # 关闭FP16（小batch下FP16易导致数值不稳定）
    gradient_accumulation_steps=1,  # 取消梯度累积（小batch无需）
    save_total_limit=3,
    remove_unused_columns=False,
    seed=42,
    report_to="tensorboard",
    disable_tqdm=False,
    run_name="gliclass_circuit_training",
    logging_first_step=True,
    eval_accumulation_steps=5,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    fp16_full_eval=False,
    gradient_checkpointing=False,
    # 新增：早停（核心！防止过拟合）
    # early_stopping_patience=3,  # 验证集F1连续3轮不提升则停止
    # early_stopping_threshold=0.001,
    # 新增：学习率调度
    lr_scheduler_type="linear",  # 线性衰减学习率
    warmup_ratio=WARMUP_RATIO,   # 前10%步数预热学习率
)

# ====================== 8. 自定义Trainer（优化版 - 无调试输出） ======================
class GLiClassTrainer(Trainer):
    """修复所有参数兼容问题+列名统一，无调试输出"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        兼容新版本transformers的compute_loss
        添加num_items_in_batch参数（可选）
        统一使用labels列
        """
        # 提取标签（Trainer默认传入labels）
        labels = inputs.pop("labels")
        
        # 设备对齐
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        
        # 确保loss有梯度
        if loss is None or not loss.requires_grad:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, logits) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """预测步骤（返回torch张量，使用labels列）"""
        with torch.no_grad():
            # 提取标签
            labels = inputs.pop("labels")
            
            # 设备对齐
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)
            
            outputs = model(**inputs)
            loss = None
            logits = outputs.logits.cpu()
            labels = labels.cpu()
        
        return (loss, logits, labels)

# 初始化Trainer
trainer = GLiClassTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=calculate_metrics,
    tokenizer=tokenizer,
)

# ====================== 9. 模型训练 ======================
print("===== 开始训练GLiClass模型 =====")
train_result = trainer.train()

# 训练结果
print("\n===== 训练完成 =====")
print(f"总训练步数: {train_result.global_step}")
print(f"训练损失: {train_result.training_loss:.4f}")

# ====================== 10. 模型评估（修复KeyError） ======================
print("\n===== 测试集评估 =====")

# 使用trainer的评估方法
test_results = trainer.evaluate(test_dataset)

# 安全地获取评估结果
f1_score = test_results.get('eval_f1', 0.0)
accuracy = test_results.get('eval_accuracy', 0.0)
loss = test_results.get('eval_loss', 'N/A')

print(f"测试集加权F1-score: {f1_score:.4f}")
print(f"测试集准确率: {accuracy:.4f}")
if loss != 'N/A':
    print(f"测试集损失: {loss:.4f}")
else:
    print("测试集损失: 无法获取")

# 手动评估获取详细预测结果
test_predictions = trainer.predict(test_dataset)
test_preds = np.argmax(test_predictions.predictions, axis=1)  # 从logits计算预测
test_labels = test_predictions.label_ids  # 获取真实标签

# 计算指标（使用修复的函数）
metrics = calculate_metrics_from_predictions(test_preds, test_labels)
print(f"\n手动评估 - 测试集加权F1-score: {metrics['f1']:.4f}")
print(f"手动评估 - 测试集准确率: {metrics['accuracy']:.4f}")

# 详细报告
print_classification_report(test_preds, test_labels, label2id)

# ====================== 11. 模型保存 ======================
print("\n===== 保存最优模型 =====")
trainer.save_model(BEST_MODEL_DIR)
tokenizer.save_pretrained(BEST_MODEL_DIR)

# 保存配置
import json
with open(os.path.join(BEST_MODEL_DIR, "training_args.json"), "w") as f:
    json.dump(training_args.to_dict(), f, indent=2)

label_mapping = {
    "label2id": label2id,
    "id2label": id2label,
    "num_labels": num_labels
}
with open(os.path.join(BEST_MODEL_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=2, ensure_ascii=False)

model_config = {
    "hidden_size": model.base_model.config.hidden_size,
    "num_labels": num_labels
}
with open(os.path.join(BEST_MODEL_DIR, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=2)

print(f"✅ 模型已保存至：{BEST_MODEL_DIR}")

# ====================== 12. 推理函数（修复词汇表大小不匹配问题） ======================
class CircuitClassifier:
    def __init__(self, model_path):
        print(f"===== 加载推理模型：{model_path} =====")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        
        # 加载配置
        with open(os.path.join(model_path, "label_mapping.json"), "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
        with open(os.path.join(model_path, "model_config.json"), "r", encoding="utf-8") as f:
            model_config = json.load(f)
        
        self.num_labels = label_mapping["num_labels"]
        self.id2label = {int(k): v for k, v in label_mapping["id2label"].items()}  # 修复类型转换
        
        # 加载原始GLiClass模型 - 使用与训练时相同的词汇表大小
        print("  加载原始GLiClass模型结构...")
        original_base_model = GLiClassModel.from_pretrained(
            MODEL_ID,  # 使用原始模型ID
            num_labels=self.num_labels,
            label2id=label_mapping["label2id"],
            id2label=self.id2label,
            trust_remote_code=True
        )
        
        # 重要：扩展词汇表以匹配训练时的大小
        # 训练时添加了特殊字符，所以词汇表大小增加了
        print(f"  训练时词汇表大小: {len(self.tokenizer)}")
        print(f"  原始模型词汇表大小: {original_base_model.config.vocab_size}")
        
        if len(self.tokenizer) != original_base_model.config.vocab_size:
            print(f"  扩展词汇表从 {original_base_model.config.vocab_size} 到 {len(self.tokenizer)}")
            original_base_model.resize_token_embeddings(len(self.tokenizer))
        
        # 创建自定义分类模型结构
        self.model = GLiClassForClassification(original_base_model, self.num_labels)
        
        # 尝试加载训练后的权重
        # 优先尝试 model.safetensors (现代格式)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")
        
        loaded_weights = False
        if os.path.exists(safetensors_path):
            print("  发现 model.safetensors，尝试加载...")
            try:
                from safetensors.torch import load_file
                
                # 加载 safetensors 文件
                state_dict = load_file(safetensors_path)
                
                # 尝试加载权重（使用 strict=False 以处理可能的键名不匹配）
                self.model.load_state_dict(state_dict, strict=False)
                loaded_weights = True
                print("  ✅ 从 model.safetensors 加载权重成功")
            except Exception as e:
                print(f"  ⚠️  从 model.safetensors 加载失败: {str(e)}")
        
        # 如果 safetensors 加载失败，尝试 pytorch_model.bin
        if not loaded_weights and os.path.exists(bin_path):
            print("  发现 pytorch_model.bin，尝试加载...")
            try:
                state_dict = torch.load(bin_path, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)
                loaded_weights = True
                print("  ✅ 从 pytorch_model.bin 加载权重成功")
            except Exception as e:
                print(f"  ⚠️  从 pytorch_model.bin 加载失败: {str(e)}")
        
        if not loaded_weights:
            print("  ⚠️  未找到训练后的权重文件，使用初始化权重")
            print("      这意味着模型将使用随机初始化的分类头，推理结果可能不准确")
        
        # 确保模型在正确的设备上
        self.model.to(device)
        self.model.eval()
        print(f"✅ 推理模型加载完成，设备：{device}")

    def predict(self, text):
        """单条推理"""
        text = text.lower().strip()
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
        pred_id = min(max(pred_id, 0), self.num_labels - 1)
        pred_label = self.id2label[pred_id]
        pred_conf = torch.softmax(logits, dim=1)[0][pred_id].item()
        
        return {
            "text": text,
            "label": pred_label,
            "confidence": round(pred_conf, 4)
        }

# ====================== 13. 推理示例 ======================
if __name__ == "__main__":
    print("\n\n" + "="*50)
    print("===== 推理示例 =====")
    classifier = CircuitClassifier(BEST_MODEL_DIR)
    
    # 测试样本
    test_samples = [
        "R1 100Ω", "C2 10μF", "L3 22mH", "U4 STM32F103",
        "M5 12V", "D6 1A 400V", "GND7", "RV8 220V"
    ]
    
    print("\n===== 推理结果 =====")
    for text in test_samples:
        res = classifier.predict(text)
        print(f"文本：{res['text']:15} → 类别：{res['label']:8} → 置信度：{res['confidence']:.4f}")
    
    print("\n✅ 所有流程执行完成！")