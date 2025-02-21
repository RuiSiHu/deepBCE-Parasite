#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author: Rui-Si Hu
# Date: 2025.2.22

import torch
import torch.nn as nn
import math


class Config:
    def __init__(self):
        self.epochs = 200
        self.classification_threshold = 0.6
        self.learning_rate = 0.00004
        self.reg = 0.001
        self.batch_size = 512
        self.num_layer = 2
        self.num_head = 8
        self.dim_embedding = 256
        self.dim_feedforward = 2048
        self.dim_k = 32
        self.dim_v = 32
        self.pooling = 'max'
        self.max_len = 77
        self.vocab_size = 24

def get_attn_pad_mask(seq, pad_token=0):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(pad_token).unsqueeze(1).unsqueeze(2).bool()
    return pad_attn_mask.expand(batch_size, 1, seq_len, seq_len)


class CustomEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.amino_acid_emb = nn.Embedding(vocab_size, d_model)
        self.positional_emb = self.create_positional_encoding(max_len, d_model)
        self.register_buffer('pe', self.positional_emb)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, input_ids):
        amino_acid_embeddings = self.amino_acid_emb(input_ids)
        seq_len = input_ids.size(1)
        positional_embeddings = self.pe[:, :seq_len, :]
        return amino_acid_embeddings + positional_embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dim_k, dim_v):
        super().__init__()
        self.d_k = dim_k
        self.d_v = dim_v
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
        V = self.W_V(V).view(batch_size, seq_len, self.n_head, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_size, self.n_head, seq_len, seq_len)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = torch.softmax(scores, dim=-1)

        context = torch.matmul(attn, V).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.n_head * self.d_v)

        output = self.fc(context)
        return self.norm(output + Q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)), attn


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class FeatureOptimizationBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dim_k, dim_v):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dim_k, dim_v)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.feature_opt_block = FeatureOptimizationBlock(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        attn_out, attn_weights = self.self_attn(x, x, x, attn_mask)
        out = self.ffn(attn_out)
        out = self.feature_opt_block(out)
        return self.norm(out + attn_out), attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dim_k, dim_v):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dim_k, dim_v)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dim_k, dim_v)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.feature_opt_block = FeatureOptimizationBlock(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, dec_input, enc_output, self_attn_mask, enc_attn_mask):
        self_attn_out, _ = self.self_attn(dec_input, dec_input, dec_input, self_attn_mask)
        enc_attn_out, attn_weights = self.enc_attn(self_attn_out, enc_output, enc_output, enc_attn_mask)
        out = self.ffn(enc_attn_out)
        out = self.feature_opt_block(out)
        return self.norm(out + enc_attn_out), attn_weights


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = CustomEmbeddingLayer(
            config.vocab_size,
            config.dim_embedding,
            config.max_len
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                config.dim_embedding,
                config.num_head,
                config.dim_feedforward,
                config.dim_k,
                config.dim_v
            ) for _ in range(config.num_layer)
        ])

    def forward(self, src, src_mask):
        x = self.embedding(src)
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, src_mask)
            attn_weights.append(attn)
        return x, attn_weights


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(
                config.dim_embedding,
                config.num_head,
                config.dim_feedforward,
                config.dim_k,
                config.dim_v
            ) for _ in range(config.num_layer)
        ])

    def forward(self, tgt, enc_output, tgt_mask, enc_mask):
        x = tgt
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, enc_output, tgt_mask, enc_mask)
            attn_weights.append(attn)
        return x, attn_weights


class EncoderDecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.pool = nn.AdaptiveMaxPool1d(1) if config.pooling == 'max' else nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(config.dim_embedding, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(64, 2)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask = get_attn_pad_mask(src)

        enc_output, enc_attn_weights = self.encoder(src, src_mask)

        if tgt is not None:
            if tgt_mask is None:
                tgt_mask = get_attn_pad_mask(tgt)
            dec_output, dec_attn_weights = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        else:
            dec_output, dec_attn_weights = enc_output, enc_attn_weights

        pooled_output = self.pool(dec_output.transpose(1, 2)).squeeze(-1)
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc_out(x)

        return output, enc_attn_weights
