#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import torch
import torch.nn as nn
import math

class Config:
    def __init__(self):
        self.vocab_size = 24  # The size of the vocabulary (can not change)
        self.dim_embedding = 256  # Residue embedding dimension (can not change)
        self.num_layer = 2  # Number of encoder/decoder layers (can not change)
        self.num_head = 8  # Number of heads in multi-head attention (can not change)
        self.dim_feedforward = 2048  # Hidden layer dimension in feedforward layer (can not change)
        self.dim_k = 32  # Embedding dimension of vector k (can not change)
        self.dim_v = 32  # Embedding dimension of vector v (can not change)
        self.max_len = 77  # Max length of input sequences (can not change)
        self.batch_size = 512  # Batch size

def get_attn_pad_mask(seq, pad_token=0):
    batch_size, seq_len = seq.size()
    return seq.eq(pad_token).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_len, seq_len)

class CustomEmbeddingLayer\
            (nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(CustomEmbeddingLayer, self).__init__()
        self.amino_acid_emb = nn.Embedding(vocab_size, d_model)
        self.positional_emb = self.create_positional_encoding(max_len, d_model)
        self.register_buffer('pe', self.positional_emb)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, input_ids):
        amino_acid_embeddings = self.amino_acid_emb(input_ids)
        seq_len = input_ids.size(1)
        positional_embeddings = self.pe[:, :seq_len, :]
        embeddings = amino_acid_embeddings + positional_embeddings
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        Q = self.W_Q(Q).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_size, self.n_head, seq_len, seq_len)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        output = self.fc(context)

        return self.norm(output + context), attn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        attn_out, _ = self.self_attn(x, x, x, attn_mask)
        out = self.ffn(attn_out)
        return self.norm(out + attn_out), _

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_ff, config):
        super(Encoder, self).__init__()
        self.embedding = CustomEmbeddingLayer(vocab_size, d_model, config.max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff) for _ in range(n_layer)])

    def forward(self, src, src_mask):
        x = self.embedding(src)
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attn_weights.append(attn)
        return x, attn_weights

class EncoderDecoderModel(nn.Module):
    def __init__(self, config):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = Encoder(config.vocab_size, config.dim_embedding, config.num_layer, config.num_head,
                               config.dim_feedforward, config)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(config.dim_embedding, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, 2)

    def forward(self, src, src_mask=None):
        if src_mask is None:
            src_mask = get_attn_pad_mask(src)

        enc_output, enc_attn_weights = self.encoder(src, src_mask)

        pooled_output = self.pool(enc_output.transpose(1, 2)).squeeze(-1)

        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc_out(x)

        return output, enc_attn_weights
