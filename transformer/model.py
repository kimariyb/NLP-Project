import torch
import torch.nn as nn

from layers import PositionalEncoding, EncoderLayer, DecoderLayer


class TransformerEncoder(nn.Module):
    """Transformer encoder 层
    编码器由多个编码器层组成，每个编码器层包含一个多头注意力层和一个前馈神经网络层。
    """
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout_rate: float,
    ):
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # 位置编码层
        self.pos_enc = PositionalEncoding(hidden_dim, dropout_rate)
        self.layers = nn.ModuleList(
            EncoderLayer(num_heads, hidden_dim, dropout_rate)
            for _ in range(num_layers)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, src_mask):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len)
            src_mask: 源序列掩码张量，形状为 (batch_size, 1, 1, seq_len)
        Returns:
            输出张量，形状为 (batch_size, seq_len, hidden_size)
        """
        # x = [batch_size, seq_len]
        # 词嵌入
        embedded = self.embedding(x)
        # embedded = [batch_size, seq_len, hidden_size]
        # 位置编码
        pos_encoded = self.pos_enc(embedded)
        # pos_encoded = [batch_size, seq_len, hidden_size]

        # 将词嵌入和位置编码相加
        x = embedded + pos_encoded
        # x = [batch_size, seq_len, hidden_size]

        # 编码器层
        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Transformer decoder 层
    解码器由多个解码器层组成，每个解码器层包含一个多头注意力层和一个前馈神经网络层。
    """
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout_rate: float,
    ):
        super(TransformerDecoder, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # 位置编码层
        self.pos_enc = PositionalEncoding(hidden_dim, dropout_rate)
        self.layers = nn.ModuleList(
            DecoderLayer(num_heads, hidden_dim, dropout_rate)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # 定义输出层
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
        # 定义softmax层
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, enc_output, tgt_mask, src_mask):
        """
        Args:
            x: 输入张量，形状为 (batch_size, tgt_seq_len)
            enc_output: 编码器输出张量，形状为 (batch_size, src_seq_len, hidden_size)
            tgt_mask: 目标序列掩码张量，形状为 (batch_size, 1, 1, seq_len)
            src_mask: 源序列掩码张量，形状为 (batch_size, 1, 1, seq_len)
        Returns:
            输出张量，形状为 (batch_size, seq_len, vocab_size)
        """
        # x = [batch_size, seq_len]
        # 词嵌入
        embedded = self.embedding(x)
        # embedded = [batch_size, seq_len, hidden_size]
        # 位置编码
        pos_encoded = self.pos_enc(embedded)
        # pos_encoded = [batch_size, seq_len, hidden_size]
        # 将词嵌入和位置编码相加
        x = embedded + pos_encoded
        # x = [batch_size, seq_len, hidden_size]

        # 解码器层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)

        # 应用归一化
        x = self.norm(x)
        # x = [batch_size, seq_len, hidden_size]

        # 输出层
        logits = self.out_proj(x)
        # logits = [batch_size, seq_len, vocab_size]

        return self.softmax(logits)


class Transformer(nn.Module):
    """Transformer 模型
    由编码器和解码器组成，用于序列到序列的任务。
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        dropout_rate: float,
    ):
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.num_layers = num_layers

        self.encoder = TransformerEncoder(
            src_vocab_size, num_layers, num_heads, hidden_dim, dropout_rate
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, num_layers, num_heads, hidden_dim, dropout_rate
        )

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: 源序列张量，形状为 (batch_size, src_seq_len)
            tgt: 目标序列张量，形状为 (batch_size, tgt_seq_len)
            src_mask: 源序列掩码张量，形状为 (batch_size, 1, 1, src_seq_len)
            tgt_mask: 目标序列掩码张量，形状为 (batch_size, 1, 1, tgt_seq_len)
        Returns:
            输出张量，形状为 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 编码器
        enc_output = self.encoder(src, src_mask)
        # enc_output = [batch_size, src_seq_len, hidden_size]
        # 解码器
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        # dec_output = [batch_size, tgt_seq_len, tgt_vocab_size]

        return dec_output
