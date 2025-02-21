#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author: Rui-Si Hu
# Date: 2025.2.22

import torch
import torch.utils.data as Data
import pickle
import logging

logging.basicConfig(level=logging.INFO)

try:
    residue2idx = pickle.load(open('./DeepModels/residue2idx.pkl', 'rb'))
    logging.info("Successfully loaded residue2idx.pkl.")
except Exception as e:
    logging.error(f"Error loading residue2idx.pkl: {e}")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_token(sequences):
    token2index = residue2idx
    token_index = []

    for seq in sequences:
        seq = list(seq)
        seq_id = [token2index.get(residue, token2index.get('X', 1)) for residue in seq]
        token_index.append(seq_id)

    return token_index


def pad_sequence(token_list, max_len):
    data = []
    for tokens in token_list:
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens.extend([0] * (max_len - len(tokens)))
        data.append(tokens)
    return data


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx].clone().detach().to(torch.long)


def construct_dataset(seq_ids, batch_size):
    seq_ids = torch.LongTensor(seq_ids)
    data_loader = Data.DataLoader(MyDataSet(seq_ids), batch_size=batch_size, shuffle=False, drop_last=False)
    return data_loader


def load_data(sequence_list, batch_size, max_len):
    token_list = transform_token(sequence_list)
    data_token = pad_sequence(token_list, max_len)
    return construct_dataset(data_token, batch_size)
