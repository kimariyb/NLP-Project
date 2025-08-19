import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple


def collect_fn(batch):
    """
    自定义的 collate 函数，用于处理不同长度的序列
    """
    # 分离输入和标签
    x, y = zip(*batch)

    # 获取每个序列的长度
    lengths = torch.tensor([len(seq) for seq in x])

    # 按长度降序排序（这对RNN处理很重要）
    lengths, sorted_idx = torch.sort(lengths, descending=True)
    x = [x[i] for i in sorted_idx]
    y = torch.tensor([y[i] for i in sorted_idx])
    
    # 填充序列
    # 注意：pad_sequence默认返回 [seq_len, batch_size, input_size]
    # 我们使用batch_first=True来得到 [batch_size, seq_len, input_size]
    padded_x = pad_sequence(x, batch_first=True)
    
    return padded_x, y, lengths


def train_fn(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: Optimizer, 
    device
) -> Tuple[float, float]:
    """
    训练模型的函数
    
    Args:
        model: 要训练的神经网络模型
        dataloader: 数据加载器，用于批量提供训练数据
        criterion: 损失函数，用于计算预测值与真实值之间的差距
        optimizer: 优化器，用于更新模型参数
        device: 训练设备（'cpu'或'cuda'），指定在哪个设备上进行计算
    """
    # 设置模型为训练模式
    model.train()

    # 初始化损失和准确率
    running_loss = 0.0
    correct = 0
    total = 0

    # 遍历数据加载器中的每个批次
    for x, y, length in dataloader:
        # 将数据移动到指定的设备上
        x, y, length = x.to(device), y.to(device), length.to(device)

        # 前向传播
        y_pred, _ = model(x, length)

        # 计算损失
        loss = criterion(y_pred, y)
        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    # 打印训练集的损失和准确率
    print(f'Train Loss: {epoch_loss}, Train Acc: {epoch_acc}')

    return epoch_loss, epoch_acc


def val_fn(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device) -> Tuple[float, float]:
    """
    验证模型的函数
    
    参数:
        model: 要验证的神经网络模型
        dataloader: 数据加载器，用于批量提供验证数据
        criterion: 损失函数，用于计算预测值与真实值之间的差距
        device: 计算设备（'cpu'或'cuda'），指定在哪个设备上进行计算
    
    返回:
        Tuple[float, float]: 包含验证损失和准确率的元组
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化损失和准确率
    running_loss = 0.0
    correct = 0
    total = 0

    # 不计算梯度
    with torch.no_grad():
        # 遍历数据加载器中的每个批次
        for x, y, length in dataloader:
            # 将数据移动到指定的设备上
            x, y, length = x.to(device), y.to(device), length.to(device)

            # 前向传播
            y_pred, _ = model(x, length)

            # 计算损失
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    # 计算平均损失和准确率
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total

    # 打印验证集的损失和准确率
    print(f'Val Loss: {epoch_loss}, Val Acc: {epoch_acc}')

    return epoch_loss, epoch_acc