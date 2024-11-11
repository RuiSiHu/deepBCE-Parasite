#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import pandas as pd
import joblib
from Feature_scripts.feature import get_proba_feature
from Feature_scripts import read_fasta_sequences
import argparse
import time
import os
import warnings

warnings.filterwarnings('ignore')

if not os.path.exists("MLResults"):
    os.makedirs("MLResults")


def predict(feature, feature_name, classifier_name, sequence_names, output_file):
    print(f"Predicting with {classifier_name} classifier......")
    scaler_path = f'./Models/{feature_name}_scaler.pkl'
    model_path = f'./Models/{feature_name}_{classifier_name}_model.pkl'
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found for feature: {feature_name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for classifier: {classifier_name} and feature: {feature_name}")

    scale = joblib.load(scaler_path)
    model = joblib.load(model_path)
    feature_scaled = scale.transform(feature)
    y_pred_prob = model.predict_proba(feature_scaled)
    y_pred = model.predict(feature_scaled)

    df_out = pd.DataFrame(columns=["Sequence_name", "Prediction", "Probability"])
    for i in range(y_pred.shape[0]):
        prediction = "Positive" if y_pred[i] == 1 else "Negative"
        probability = f"{y_pred_prob[i, 1] * 100:.2f}%" if y_pred[i] == 1 else f"{y_pred_prob[i, 0] * 100:.2f}%"
        df_out = pd.concat([df_out, pd.DataFrame({
            "Sequence_name": [sequence_names[i]],
            "Prediction": [prediction],
            "Probability": [probability]
        })], ignore_index=True)

    df_out.to_excel(output_file, index=False)
    print("Job finished! Results saved to:", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.",
                                     description="Sequence-based prediction of B-cell epitope in parasite with feature representation learning")
    parser.add_argument("-i", required=True, default=None, help="input fasta file")
    parser.add_argument("-f", required=True,
                        choices=["AAC", "ASDC", "CKSAAGP", "CKSAAP", "DDE", "DPC", "GAAC", "GDPC", "GTPC", "PAAC",
                                 "QSOrder", "SOCNumber"], help="feature type")
    parser.add_argument("-c", required=True, choices=["GNB", "LGBM", "RF", "SVM"], help="classifier type")
    args = parser.parse_args()

    classifier_mapping = {
        "GNB": "GaussianNB",
        "LGBM": "LGBMClassifier",
        "RF": "RandomForestClassifier",
        "SVM": "SVC"
    }

    time_start = time.time()
    print("Sequence checking......")
    sequences, sequence_names = read_fasta_sequences.read_protein_sequences(args.i)

    print(f"Sequence encoding using {args.f} feature......")
    encodings = get_proba_feature(sequences, args.f)

    classifier_full_name = classifier_mapping[args.c]
    output_file = f"MLResults/{args.f}_{args.c}_results.xlsx"

    print(f"Predicting with {args.c} classifier......")
    predict(encodings, args.f, classifier_full_name, sequence_names, output_file)

    time_end = time.time()
    print('Total time cost:', time_end - time_start, 'seconds')
