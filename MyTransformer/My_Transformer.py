"""
The original source code comes from https://zhuanlan.zhihu.com/p/690355021
I made some modifications to it.
"""

import numpy as np
import torch
from torch import nn
import math

import Utility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=4):
        """
        d_model: dimension of Embeddings and Input/Output
        num_heads: h
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divided by h"
        self.Wv = nn.Linear(d_model, d_model, bias=False).to(device)  # for Value
        self.Wk = nn.Linear(d_model, d_model, bias=False).to(device)  # for Key
        self.Wq = nn.Linear(d_model, d_model, bias=False).to(device)  # for Query
        self.Wo = nn.Linear(d_model, d_model, bias=False).to(device)  # for Linear


@Utility.add_to_class(MultiHeadAttention)
def check_sdpa_inputs(self, x):
    assert x.size(
        1) == self.num_heads, f"The expected size of x is({-1, self.num_heads, -1, self.d_model // self.num_heads}), but obtain {x.size()}"
    assert x.size(3) == self.d_model // self.num_heads


@Utility.add_to_class(MultiHeadAttention)
def scaled_dot_product_attention(
        self, query, key, value, attention_mask=None, key_padding_mask=None):
    """
    query : shape=(batch_size, num_heads, query_sequence_length, d_model//num_heads)
    key : shape=(batch_size, num_heads, key_sequence_length, d_model//num_heads)
    value : shape=(batch_size, num_heads, key_sequence_length, d_model//num_heads)
    attention_mask : shape=(query_sequence_length, key_sequence_length)
    key_padding_mask : shape=(sequence_length, key_sequence_length)
    """
    self.check_sdpa_inputs(query)
    self.check_sdpa_inputs(key)
    self.check_sdpa_inputs(value)

    d_k = query.size(-1)
    tgt_len, src_len = query.size(-2), key.size(-2)

    # logits = (B, H, tgt_len, E) * (B, H, E, src_len) = (B, H, tgt_len, src_len)
    logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 注意力遮罩
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            assert attention_mask.size() == (tgt_len, src_len)
            attention_mask = attention_mask.unsqueeze(0)
            logits = logits + attention_mask
        else:
            raise ValueError(f"mask size = {attention_mask.size()}")

    # key mask
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        logits = logits + key_padding_mask

    attention = torch.softmax(logits, dim=-1)
    output = torch.matmul(attention, value)  # output shape: (batch_size, num_heads, sequence_length, d_model)

    return output, attention


@Utility.add_to_class(MultiHeadAttention)
def split_into_heads(self, x, num_heads):
    batch_size, seq_length, d_model = x.size()
    x = x.view(batch_size, seq_length, num_heads, d_model // num_heads)
    return x.transpose(1, 2)  # output shape: (batch_size, num_heads, seq_length, d_model // num_heads)


@Utility.add_to_class(MultiHeadAttention)
def combine_heads(self, x):
    batch_size, num_heads, seq_length, head_hidden_dim = x.size()
    return x.transpose(1, 2).contiguous().view(
        batch_size, seq_length, num_heads * head_hidden_dim)


@Utility.add_to_class(MultiHeadAttention)
def forward(self, q, k, v, attention_mask=None, key_padding_mask=None):
    """
    q : shape=(batch_size, query_sequence_length, d_model)
    k : shape=(batch_size, key_sequence_length, d_model)
    v : shape=(batch_size, key_sequence_length, d_model)
    attention_mask : shape=(query_sequence_length, key_sequence_length)
    key_padding_mask : shape=(sequence_length, key_sequence_length)
    """
    q = self.Wq(q)
    k = self.Wk(k)
    v = self.Wv(v)

    q = self.split_into_heads(q, self.num_heads)
    k = self.split_into_heads(k, self.num_heads)
    v = self.split_into_heads(v, self.num_heads)

    attn_values, attn_weights = self.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attention_mask=attention_mask,
        key_padding_mask=key_padding_mask,
    )
    grouped = self.combine_heads(attn_values)
    output = self.Wo(grouped)

    self.attention_weigths = attn_weights

    print('output.shape: ', output.shape)
    return output


def generate_square_subsequent_mask(size: int):
    mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(hidden_dim=n_dim, num_heads=n_heads)
        self.norm1 = nn.LayerNorm(n_dim)
        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm2 = nn.LayerNorm(n_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_padding_mask=None):
        assert x.ndim == 3, "Expected input to be 3-dim, got {}".format(x.ndim)
        att_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm1(att_output))

        ff_output = self.ff(x)
        output = x + self.norm2(ff_output)

        return output


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            n_dim: int,
            dropout: float,
            n_encoder_blocks: int,
            n_heads: int):
        super(Encoder, self).__init__()
        self.n_dim = n_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_dim
        )
        self.positional_encoding = PositionalEncoding(
            d_model=n_dim,
            dropout=dropout
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_dim, dropout, n_heads) for _ in range(n_encoder_blocks)
        ])

    def forward(self, x, padding_mask=None):
        x = self.embedding(x) * math.sqrt(self.n_dim)
        x = self.positional_encoding(x)
        for block in self.encoder_blocks:
            x = block(x=x, src_padding_mask=padding_mask)
        return x


if __name__ == '__main__':
    print('test1: MultiHeadAttention test')

    d_model = 512
    h = 8
    attention = MultiHeadAttention(d_model=d_model, num_heads=h)

    q = torch.rand([32, 128, 512]).to(device)
    k = torch.rand([32, 256, 512]).to(device)
    v = torch.rand([32, 256, 512]).to(device)

    data = attention(q, k, v)

    print('test2: generate_square_subsequent_mask test')

    mask = generate_square_subsequent_mask(d_model // h)
    print('mask:', mask)

'''
如何生成 Q、K、V
通过线性变换，每个词都会生成三个表示：Q (Query)、K (Key)、V (Value)。
我们假设经过计算得到如下向量（实际是矩阵运算的结果）：

词语	Q（Query，需求）	K（Key，身份）	V（Value，信息）
猫	[0.1, 0.3]	[0.5, 0.2]	[0.7, 0.6]
喜欢	[0.2, 0.4]	[0.1, 0.5]	[0.8, 0.9]
鱼	[0.3, 0.1]	[0.4, 0.3]	[0.5, 0.7]
'''
