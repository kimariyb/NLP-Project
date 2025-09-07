import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


def train_fn(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optimizer, device) -> float:
    """
    该函数用于训练模型，通过迭代数据加载器中的批次数据，计算损失并更新模型参数。

    Args:
        model: 要训练的神经网络模型
        loader: 数据加载器，用于批量提供训练数据
        criterion: 损失函数，用于计算预测值与真实值之间的差距
        optimizer: 优化器，用于更新模型参数
        device: 训练设备（'cpu'或'cuda'），指定在哪个设备上进行计算

    Returns:
        float: 平均损失
    """
    # 设置模型为训练模式
    model.train()

    # 初始化损失
    total_loss = 0

    # 遍历数据加载器中的每个批次
    for batch in tqdm(loader, desc='train loop', total=len(loader), unit='batch'):
        src, tgt, _, _, _ = batch
        src = src.to(device)
        tgt = tgt.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 向前传播
        output = model(src, tgt)

        output_dim = output.shape[-1]
        # 对于 batch_first=True，输出形状是 [batch_size, seq_len, output_dim]
        output = output[:, 1:].contiguous().view(-1, output_dim)  # 跳过第一个token
        tgt = tgt[:, 1:].contiguous().view(-1)  # 跳过第一个token

        loss = criterion(output, tgt)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def val_fn(model: nn.Module, loader: DataLoader, criterion: nn.Module, device) -> float:
    """
    验证模型的函数

    参数:
        model: 要验证的神经网络模型
        loader: 数据加载器，用于批量提供验证数据
        criterion: 损失函数，用于计算预测值与真实值之间的差距
        device: 计算设备（'cpu'或'cuda'），指定在哪个设备上进行计算

    返回:
        Tuple[float, float]: 包含验证损失和准确率的元组
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化损失
    total_loss = 0

    # 遍历数据加载器中的每个批次
    for batch in tqdm(loader, desc='val loop', total=len(loader), unit='batch'):
        # 解包数据
        src, tgt, _, _, _ = batch
        src = src.to(device)
        tgt = tgt.to(device)

        with torch.no_grad():
            output = model(src, tgt, 0)

        output_dim = output.shape[-1]
        # 对于 batch_first=True，输出形状是 [batch_size, seq_len, output_dim]
        output = output[:, 1:].contiguous().view(-1, output_dim)  # 跳过第一个token
        tgt = tgt[:, 1:].contiguous().view(-1)  # 跳过第一个token

        loss = criterion(output, tgt)
        total_loss += loss.item()

    return total_loss / len(loader)
