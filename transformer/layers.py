import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEncoding(nn.Module):
    """
    三角函数位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i / hidden_size))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / hidden_size))

    Args:
        hidden_size: 模型维度
        dropout_rate: dropout 率
        max_len: 最大序列长度
    """
    def __init__(self, hidden_size, dropout_rate, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        # 位置编码矩阵
        # pe = [max_len, hidden_size]
        pe = torch.zeros(max_len, hidden_size)

        # position = [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = [hidden_size / 2]
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() \
                * (-torch.log(torch.tensor(1e4)) / hidden_size)
            )
        # pe = [max_len, hidden_size]
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数位置
        # pe = [1, max_len, hidden_size]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
        Returns:
            输出张量，形状为 (batch_size, seq_len, hidden_size)
        """
        # x = [batch_size, seq_len, hidden_size]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, num_heads, hidden_size, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_size = hidden_size // num_heads

        # 增加线性变换层
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        self.init_weights()

    def init_weights(self):
        # 使用 xavier_uniform_ 初始化线性变换层
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: 查询张量，形状为 (batch_size, seq_len, hidden_size)
            k: 键张量，形状为 (batch_size, seq_len, hidden_size)
            v: 值张量，形状为 (batch_size, seq_len, hidden_size)
            mask: 可选的掩码张量，形状为 (batch_size, 1, 1, seq_len)
        Returns:
            输出张量，形状为 (batch_size, seq_len, hidden_size)
        """
        # q = [batch_size, seq_len, hidden_size]
        q = self.q_proj(q)
        # k = [batch_size, seq_len, hidden_size]
        k = self.k_proj(k)
        # v = [batch_size, seq_len, hidden_size]
        v = self.v_proj(v)

        batch_size = q.size(0)

        # 分头
        # q = [batch_size, seq_len, num_heads, head_size]
        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        # k = [batch_size, seq_len, num_heads, head_size]
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        # v = [batch_size, seq_len, num_heads, head_size]
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        # 计算注意力分数
        # scores = [batch_size, num_heads, seq_len, seq_len]
        # scores = (q @ k.transpose(-2, -1) / d_k) * k
        scores = torch.matmul(
            q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        # attn_output = [batch_size, num_heads, seq_len, head_size]
        attn_output = torch.matmul(attn_weights, v)

        # 合并头
        # attn_output = [batch_size, seq_len, hidden_size]
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')

        # 输出层
        # output = [batch_size, seq_len, hidden_size]
        output = self.out_proj(attn_output)

        return output


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, num_heads, hidden_dim, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(num_heads, hidden_dim, dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            mask: 可选的掩码张量，形状为 (batch_size, 1, 1, seq_len)
        Returns:
            输出张量，形状为 (batch_size, seq_len, hidden_size)
        """
        # 多头自注意力层
        attn_output = self.attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # 前馈网络
        ffn_output = self.ffn(x)
        # 残差连接和归一化
        x  = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, num_heads, hidden_dim, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(num_heads, hidden_dim, dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(hidden_dim)
        self.dropout2 = nn.Dropout(hidden_dim)
        self.dropout3 = nn.Dropout(hidden_dim)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, hidden_size)
            enc_output: 编码器输出张量，形状为 (batch_size, seq_len, hidden_size)
            src_mask: 可选的源掩码张量，形状为 (batch_size, 1, 1, seq_len)
            tgt_mask: 可选的目标掩码张量，形状为 (batch_size, 1, 1, seq_len)
        Returns:
            输出张量，形状为 (batch_size, seq_len, hidden_size)
        """
        # 掩码多头自注意力
        mask_attn = self.attn(x, x, x, tgt_mask)
        x = x + self.dropout1(mask_attn)
        x = self.norm1(x)

        # 多头自注意力
        # encoder 层的输出作为 k 和 v
        attn_output = self.attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)

        # 前馈神经网络
        ffn_output = self.ffn(x)
        x = x + self.dropout3(ffn_output)
        x = self.norm3(x)

        return x


if __name__ == '__main__':
    # 测试多头注意力机制
    num_heads = 2
    hidden_size = 64
    dropout_rate = 0.1
    seq_len = 10
    batch_size = 2

    # 随机生成输入张量
    q = torch.randn(batch_size, seq_len, hidden_size)
    k = torch.randn(batch_size, seq_len, hidden_size)
    v = torch.randn(batch_size, seq_len, hidden_size)

    # 初始化多头注意力层
    mha = MultiHeadAttention(num_heads, hidden_size, dropout_rate)

    # 前向传播
    output = mha(q, k, v)

    print("输入张量形状:", q.shape)
    print("输出张量形状:", output.shape)
