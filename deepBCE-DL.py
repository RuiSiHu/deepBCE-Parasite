#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author: Rui-Si Hu
# Date: 2025.2.22

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import time

from preprocess.data_process import load_data
from DeepModels.deepBCE_model import EncoderDecoderModel, Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_folder = 'DeepResults'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def get_attn_pad_mask(seq, pad_token=0):
    batch_size, seq_len = seq.size()
    return seq.eq(pad_token).unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_len, seq_len)

def model_eval(test_data, model, classification_threshold=0.6):
    label_pred = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            inputs = batch.to(device)
            src_mask = get_attn_pad_mask(inputs)

            enc_output, _ = model.encoder(inputs, src_mask)
            pooled_output = model.pool(enc_output.transpose(1, 2)).squeeze(-1)
            x = model.fc1(pooled_output)
            x = model.relu(x)
            x = model.dropout(x)
            outputs = model.fc_out(x)

            pred_prob_all = torch.softmax(outputs, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_class = (pred_prob_positive > classification_threshold).float()

            label_pred = torch.cat([label_pred, pred_class])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

    return label_pred.cpu().numpy(), pred_prob.cpu().numpy()

def load_text_file(fasta_file):
    with open(fasta_file) as f:
        lines = f.read().split('>')
        ids = []
        sequences = []
        for record in lines[1:]:
            split_record = record.splitlines()
            seq_id = split_record[0].strip()
            sequence = ''.join(split_record[1:]).upper()
            ids.append(seq_id)
            sequences.append(sequence)
        return ids, sequences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Sequence-Based Prediction of B-cell Epitope in Human and Veterinary Parasites Using Transformer-based Deep Learning")
    parser.add_argument("-i", required=True, help="input fasta file")
    parser.add_argument("-o", default="DeepResults.csv", help="output a CSV results file")

    args = parser.parse_args()

    time_start = time.time()

    print("Data loading......")
    seq_ids, sequences_list = load_text_file(args.i)

    print("Data processing......")
    config = Config()
    test_loader = load_data(sequences_list, config.batch_size, config.max_len)

    print("Model loading......")
    model = EncoderDecoderModel(config).to(device)

    try:
        state_dict = torch.load("./DeepModels/deepBCE_model.pkl", map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        sys.exit(1)

    print("Predicting......")
    y_pred, y_pred_prob = model_eval(test_loader, model)

    results = pd.DataFrame(columns=["Seq_ID", "Sequences", "Prediction", "Confidence"])
    result_list = []

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            y_prob = f"{round(y_pred_prob[i] * 100, 2)}%"
            result_list.append({
                "Seq_ID": seq_ids[i],
                "Sequences": ''.join(sequences_list[i]),
                "Prediction": "Positive",
                "Confidence": y_prob
            })
        else:
            y_prob = f"{round((1 - y_pred_prob[i]) * 100, 2)}%"
            result_list.append({
                "Seq_ID": seq_ids[i],
                "Sequences": ''.join(sequences_list[i]),
                "Prediction": "Negative",
                "Confidence": y_prob
            })

    results = pd.concat([pd.DataFrame(result_list)], ignore_index=True)

    results.to_csv(os.path.join(result_folder, args.o), index=False)

    time_end = time.time()
    print("Job finished! Total time:", time_end - time_start, "seconds")
