import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class EncoderRNN(nn.Module):
    """GRU 编码层"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # GRU 层
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 嵌入层
        # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch_size, src_len, embed_size]

        # GRU 层
        # output = [batch_size, src_len, hidden_size]
        # hidden = [num_layers, batch_size, hidden_size]
        output, hidden = self.gru(embedded)

        return output, hidden


class DecoderRNN(nn.Module):
    """GRU 解码层"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout=0.2):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # GRU 层
        self.gru = nn.GRU(
            embed_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出层
        self.out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 嵌入层
        # input = [batch_size, 1]
        # hidden = [num_layers, batch_size, hidden_size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch_size, 1, embed_size]

        # GRU 层
        # output = [batch_size, seq_len, hidden_size]
        # hidden = [num_layers, batch_size, hidden_size]
        output, hidden = self.gru(embedded, hidden)

        # 输出层
        # prediction = [batch_size, vocab_size]
        prediction = self.out(output.squeeze(1))
        # 使用 softmax 函数
        prediction = F.softmax(prediction, dim=1)

        return prediction, hidden


class NormalSeq2Seq(nn.Module):
    """基于 GRU 的 Seq2Seq 模型"""
    def __init__(
        self,
        input_word_size: int,
        output_word_size: int,
        encoder_dim: int,
        decoder_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super(NormalSeq2Seq, self).__init__()
        self.encoder = EncoderRNN(
            vocab_size=input_word_size,
            embed_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = DecoderRNN(
            vocab_size=output_word_size,
            embed_size=decoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        # src = [batch_size, src_len]
        # tgt = [batch_size, tgt_len]
        # teacher_forcing_ratio = 0.5
        if tgt is not None: # 对应训练任务
            batch_size = tgt.shape[0]
            tgt_len = tgt.shape[1]
            tgt_vocab_size = self.decoder.vocab_size
            # 用于存储 Decoder 结果的张量
            # outputs = [batch_size, tgt_len, tgt_vocab_size]
            outputs = torch.zeros(batch_size, tgt_len, int(tgt_vocab_size)).to(src.device)

            # 编码器的 hidden 状态作为解码器的初始状态
            # encoder_outputs = [batch_size, src_len, hidden_size]
            # encoder_hidden = [num_layers, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(src)

            # 第一个输入是 <SOS>
            # input = [batch_size, 1]
            input = tgt[:, 0].unsqueeze(1)

            # 解码
            for t in range(1, tgt_len):
                # output = [batch_size, 1, tgt_vocab_size]
                output, hidden = self.decoder(input, encoder_hidden)
                # 把解码器输出放入 output
                outputs[:, t, :] = output
                # 随机决定使用 teacher forcing
                teacher_force = random.random() < teacher_forcing_ratio
                # 找出最大概率输出
                top1 = output.argmax(1)
                # 如果使用 teacher forcing，下一个输入是目标序列的下一个 token
                # 否则，下一个输入是上一个时间步的预测结果
                input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        else:
            # 对应推理任务
            batch_size = src.shape[0]
            tgt_vocab_size = self.decoder.vocab_size
            # 存储解码器结果的列表
            outputs = []
            _, hidden = self.encoder(src)
            # 第一个输入是 <SOS> (假设索引为0)
            input = torch.zeros(batch_size, 1, dtype=torch.long, device=src.device)
            max_length = 50  # 设置最大生成长度，避免无限循环
            for _ in range(max_length):
                output, hidden = self.decoder(input, hidden)
                # 把解码器输出放入 outputs
                outputs.append(output)
                # 找出最大概率输出
                top1 = output.argmax(1)
                # 如果所有样本都预测到 EOS 则停止
                if (top1 == 1).all():
                    break
                # 否则，下一个输入是上一个时间步的预测结果
                input = top1.unsqueeze(1)

            # 将输出列表转换为张量
            outputs = torch.stack(outputs, dim=1).squeeze(2)

        return outputs

