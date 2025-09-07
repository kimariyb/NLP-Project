import os
import random
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import train_fn, val_fn
from model import NormalSeq2Seq
from dataset import create_dataloader

# 超参数
EPOCHS = 100
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
TEACHER_FORCING_RATIO = 0.5

NUM_LAYERS = 2
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.5

MODEL_NAME = 'Normal'

# 设置随机种子以确保结果的可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 设置设备
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # 打印将要训练用的设备
    print(f"Using device: {DEVICE}")

    loader, dataset = create_dataloader(
        data_path='./data/eng-fra-v2.txt',
        batch_size=BATCH_SIZE,
        shuffle=True,
        max_length=10,
        split_ratio=0.8,
    )

    train_loader, val_loader = loader

    # 创建模型
    model = NormalSeq2Seq(
        input_word_size=dataset.src_vocab.n_words,
        output_word_size=dataset.tgt_vocab.n_words,
        encoder_dim=EMBEDDING_DIM,
        decoder_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_ID)

    # 初始化日志文件
    os.makedirs('./logs', exist_ok=True)
    log_file = f'./logs/{MODEL_NAME}_log.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")

    # 训练模型
    for epoch in tqdm(
        range(EPOCHS),
        desc=f"Training Epoch {EPOCHS}",
        total=EPOCHS,
        unit="epoch",
    ):
        # 首先训练模型
        train_loss = train_fn(model, train_loader, criterion, optimizer, DEVICE)
        # 然后验证模型
        val_loss = val_fn(model, val_loader, criterion, DEVICE)

        # 将训练和验证结果写入日志
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_loss:.4f}\n")
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 每隔 10 轮保存一次模型
        if (epoch+1) % 10 == 0:
            torch.save(
                model.state_dict(),
                f"./logs/{MODEL_NAME}_epoch_{epoch+1}_val_{val_loss:.4f}.pth"
            )


if __name__ == '__main__':
    main()

