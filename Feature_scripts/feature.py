#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import pandas as pd
import os
from Feature_scripts.AAC import get_AAC
from Feature_scripts.ASDC import get_ASDC
from Feature_scripts.CKSAAGP import get_CKSAAGP
from Feature_scripts.CKSAAP import get_CKSAAP
from Feature_scripts.DDE import get_DDE
from Feature_scripts.DPC import get_DPC
from Feature_scripts.GAAC import get_GAAC
from Feature_scripts.GDPC import get_GDPC
from Feature_scripts.GTPC import get_GTPC
from Feature_scripts.PAAC import get_PAAC
from Feature_scripts.QSOrder import get_QSOrder
from Feature_scripts.SOCNumber import get_SOCNumber

FEATURE_EXTRACTORS = {
    "AAC": get_AAC,
    "ASDC": get_ASDC,
    "CKSAAGP": get_CKSAAGP,
    "CKSAAP": get_CKSAAP,
    "DDE": get_DDE,
    "DPC": get_DPC,
    "GAAC": get_GAAC,
    "GDPC": get_GDPC,
    "GTPC": get_GTPC,
    "PAAC": get_PAAC,
    "QSOrder": get_QSOrder,
    "SOCNumber": get_SOCNumber,
}

FEATURE_SELECTION_MAPPING_FILE = "./Feature_scripts/feature_selection_mapping.csv"


def remove_non_numeric_columns(df):
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        raise ValueError(
            "All columns were removed from the DataFrame. Check if the input data has valid numeric columns."
        )
    return numeric_df



def apply_feature_selection(df, feature_type):
    if not os.path.exists(FEATURE_SELECTION_MAPPING_FILE):
        raise FileNotFoundError(f"Feature selection mapping file not found: {FEATURE_SELECTION_MAPPING_FILE}")

    mapping_df = pd.read_csv(FEATURE_SELECTION_MAPPING_FILE, header=None)
    mapping_df.columns = ['Feature_Type'] + [f'Feature_{i}' for i in range(1, mapping_df.shape[1])]

    feature_row = mapping_df[mapping_df['Feature_Type'] == feature_type]
    if feature_row.empty:
        raise ValueError(f"Feature type {feature_type} not found in feature selection mapping file.")

    selected_columns = feature_row.values[0][1:]
    selected_columns = [
        int(col) for col in selected_columns if pd.notna(col) and int(col) < df.shape[1]
    ]

    return df.iloc[:, selected_columns]


def get_proba_feature(fastas, feature_type):
    if feature_type not in FEATURE_EXTRACTORS:
        raise ValueError(f"Feature type {feature_type} is not supported.")


    feature_extractor = FEATURE_EXTRACTORS[feature_type]
    feature = feature_extractor(fastas)

    feature_df = pd.DataFrame(feature[1:], columns=feature[0])

    numeric_feature_df = remove_non_numeric_columns(feature_df)

    selected_features = apply_feature_selection(numeric_feature_df, feature_type)

    return selected_features
