import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x * math.sqrt(x.size(-1)) + self.pe[:, :x.size(1)]

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        output, _ = self.attention(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attention = AttentionBlock(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_padding_mask):
        x = self.norm1(x + self.dropout(self.attention(x, x, x, key_padding_mask=src_padding_mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = AttentionBlock(d_model, num_heads, dropout)
        self.cross_attention = AttentionBlock(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, encoder_output, tgt_mask, src_padding_mask, tgt_padding_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, key_padding_mask=tgt_padding_mask, attn_mask=tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, encoder_output, encoder_output, key_padding_mask=src_padding_mask)))
        return self.norm3(x + self.dropout(self.ffn(x)))

class TransformerNMT(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=4, num_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
    def forward(self, src, tgt):
        src_padding_mask = src == 0
        tgt_padding_mask = tgt == 0
        length = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(length, length, device=tgt.device, dtype=torch.bool), diagonal=1)
        src = self.pos_encoding(self.src_embedding(src))
        tgt = self.pos_encoding(self.tgt_embedding(tgt))
        for layer in self.encoder_layers:
            src = layer(src, src_padding_mask)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask, src_padding_mask, tgt_padding_mask)
        return self.output_layer(tgt)
