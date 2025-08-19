import os
import torch
import torch.nn as nn

from datasets import NameClassDataset, ONEHOT_DIM, CATEGORIES_DIM
from models import MyRNN, MyLSTM, MyGRU
from train import train_fn, val_fn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm


# 自定义超参数
EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MODEL_NAME = 'RNN'
HIDDEN_DIM = 128
NUM_LAYERS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # 打印将要训练用的的设备
    print(f"Using device: {DEVICE}")

    # 首先读取数据集
    x_list, y_list = NameClassDataset.read_data("./data/name_classfication.txt")
    dataset = NameClassDataset(
        x=x_list, y=y_list
    )

    # 将数据集划分为训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
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

    # 实例化损失函数
    criterion = nn.NLLLoss()

    # 创建日志目录
    os.makedirs('./logs', exist_ok=True)

    # 初始化日志文件
    log_file = f'./logs/{MODEL_NAME}_log.csv'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    # 训练模型
    for epoch in tqdm(
        range(EPOCHS),
        desc=f"Training {MODEL_NAME} model",
        unit="epoch",
        total=EPOCHS,
    ):
        # 首先训练模型
        train_loss, train_acc = train_fn(
            model, train_dataloader, optimizer, criterion, DEVICE
        )
        # 然后验证模型
        val_loss, val_acc = val_fn(model, val_dataloader, criterion, DEVICE)

        # 将训练和验证结果写入日志
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_acc:.4f},{val_loss:.4f},{val_acc:.4f}\n")
            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 每隔 10 轮保存一次模型
        if (epoch+1) % 10 == 0:
            torch.save(
                model.state_dict(),
                f"./logs/{MODEL_NAME}_epoch_{epoch+1}_acc_{val_acc:.4f}.pth"
            )


if __name__ == '__main__':
    main()