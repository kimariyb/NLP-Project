import os
import platform
import random
from collections import defaultdict

import torch
import torch.nn as nn

from datasets import NameClassDataset, ONEHOT_DIM, CATEGORIES_DIM, CATEGORIES
from models import MyRNN, MyLSTM, MyGRU
from train import train_fn, val_fn, collect_fn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm


# 自定义超参数
EPOCHS = 100
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
MODEL_NAME = os.environ.get('MODEL_NAME', 'RNN')
HIDDEN_DIM = 128
NUM_LAYERS = 1
SEED = 42                 # 固定随机种子，保证可复现
TRAIN_RATIO = 0.8         # 训练集比例
# macOS 上多进程 DataLoader 易卡死/报错且收益有限，默认关闭；其他平台用 8
NUM_WORKERS = 0 if platform.system() == 'Darwin' else 8

# 设置计算设备：优先 CUDA，其次 Apple Silicon 的 MPS，最后 CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def stratified_split(dataset, y_list, train_ratio=0.8, seed=42):
    """
    按类别分层划分训练集/验证集：每个类别内部按相同比例切分，
    保证稀有类别在训练集和验证集里都出现（避免 random_split 把少数类切没）。
    """
    rng = random.Random(seed)
    class_to_idx = defaultdict(list)
    for i, y in enumerate(y_list):
        class_to_idx[y].append(i)

    train_idx, val_idx = [], []
    for y, idxs in class_to_idx.items():
        rng.shuffle(idxs)
        n_train = int(round(train_ratio * len(idxs)))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def compute_class_weights(y_list):
    """
    计算类别权重（按频率反比），缓解 129:1 的极端不平衡。
    返回归一化后的权重：mean(weight) = 1，稀有类权重更高。
    """
    labels = torch.tensor([CATEGORIES.index(y) for y in y_list], dtype=torch.long)
    counts = torch.bincount(labels, minlength=CATEGORIES_DIM).float()
    weights = 1.0 / counts
    weights = weights / weights.sum() * CATEGORIES_DIM
    return weights


def predict_all(model, dataloader, device):
    """在 dataloader 上跑推理，返回 (all_labels, all_preds) 的 python list。"""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for x, y, length in dataloader:
            x, y, length = x.to(device), y.to(device), length.to(device)
            logits = model(x, length)
            preds = logits.argmax(dim=1)
            all_labels.extend(y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    return all_labels, all_preds


def main():
    # 固定随机种子，保证划分/训练可复现
    torch.manual_seed(SEED)
    random.seed(SEED)
    if DEVICE.type == 'mps':
        torch.mps.manual_seed(SEED)

    # 打印将要训练用的设备
    print(f"Using device: {DEVICE}")

    # 首先读取数据集
    x_list, y_list = NameClassDataset.read_data("./data/name_classfication.txt")
    dataset = NameClassDataset(x=x_list, y=y_list)

    # 分层划分训练集/验证集（按类别比例，固定种子）
    train_dataset, val_dataset = stratified_split(
        dataset, y_list, train_ratio=TRAIN_RATIO, seed=SEED
    )
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collect_fn,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collect_fn,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # 实例化模型
    if MODEL_NAME == 'RNN':
        model = MyRNN(
            input_size=ONEHOT_DIM, hidden_size=HIDDEN_DIM,
            output_size=CATEGORIES_DIM, num_layers=NUM_LAYERS
        ).to(DEVICE)
    elif MODEL_NAME == 'LSTM':
        model = MyLSTM(
            input_size=ONEHOT_DIM, hidden_size=HIDDEN_DIM,
            output_size=CATEGORIES_DIM, num_layers=NUM_LAYERS
        ).to(DEVICE)
    elif MODEL_NAME == 'GRU':
        model = MyGRU(
            input_size=ONEHOT_DIM, hidden_size=HIDDEN_DIM,
            output_size=CATEGORIES_DIM, num_layers=NUM_LAYERS
        ).to(DEVICE)

    # 实例化优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 实例化损失函数：加入类别权重，缓解类别不平衡
    class_weights = compute_class_weights(y_list).to(DEVICE)
    criterion = nn.NLLLoss(weight=class_weights)

    # 创建日志目录（每个模型一个子目录，与 plot.py 读取路径保持一致）
    model_dir = f'./logs/{MODEL_NAME}'
    os.makedirs(model_dir, exist_ok=True)

    # 初始化日志文件（每次运行覆盖，避免重跑追加导致重复行）
    log_file = f'{model_dir}/{MODEL_NAME}_log.csv'
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    # 最佳模型存档（仅做 checkpoint，不做早停；训练固定跑满 EPOCHS）
    best_val_acc = -1.0
    best_epoch = 0
    best_path = f"{model_dir}/{MODEL_NAME}_best.pth"

    # 训练模型（无早停，固定训练 EPOCHS 轮）
    for epoch in tqdm(
        range(EPOCHS),
        desc=f"Training {MODEL_NAME} model",
        unit="epoch",
        total=EPOCHS,
    ):
        # 首先训练模型
        train_loss, train_acc = train_fn(
            model, train_dataloader, criterion, optimizer, DEVICE
        )
        # 然后验证模型
        val_loss, val_acc = val_fn(model, val_dataloader, criterion, DEVICE)

        # 将训练和验证结果写入日志
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 验证集准确率创新高则保存最佳权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best val_acc {val_acc:.4f}, saved to {best_path}")

    print(f"Training done. Best val_acc={best_val_acc:.4f} @ epoch {best_epoch}")

    # ===== sklearn 指标评估（验证集上，用最佳 checkpoint）=====
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    labels, preds = predict_all(model, val_dataloader, DEVICE)

    acc = accuracy_score(labels, preds)
    prec_w = precision_score(labels, preds, average="weighted", zero_division=0)
    prec_m = precision_score(labels, preds, average="macro", zero_division=0)
    rec_w = recall_score(labels, preds, average="weighted", zero_division=0)
    rec_m = recall_score(labels, preds, average="macro", zero_division=0)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, target_names=CATEGORIES, zero_division=0)

    print(f"\n===== Sklearn metrics ({MODEL_NAME}, best @ ep{best_epoch}) =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: weighted={prec_w:.4f}  macro={prec_m:.4f}")
    print(f"Recall   : weighted={rec_w:.4f}  macro={rec_m:.4f}")
    print(f"F1       : weighted={f1_w:.4f}  macro={f1_m:.4f}")
    print("Classification report:")
    print(report)

    # 持久化到日志目录，便于三模型横向对比
    with open(f"{model_dir}/{MODEL_NAME}_sklearn.txt", "w") as f:
        f.write(f"MODEL: {MODEL_NAME}\n")
        f.write(f"Best val_acc @ epoch: {best_epoch} ({best_val_acc:.4f})\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: weighted={prec_w:.4f}  macro={prec_m:.4f}\n")
        f.write(f"Recall   : weighted={rec_w:.4f}  macro={rec_m:.4f}\n")
        f.write(f"F1       : weighted={f1_w:.4f}  macro={f1_m:.4f}\n")
        f.write("\nClassification report:\n")
        f.write(report)


if __name__ == '__main__':
    main()
