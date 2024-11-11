#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re
import os
import platform
import numpy as np

def get_SOCNumber(sequences, nlag=5):

    dataFile = os.path.join('Feature_scripts', 'Schneider-Wrede.txt')
    dataFile1 = os.path.join('Feature_scripts', 'Grantham.txt')

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AA1 = 'ARNDCQEGHILKMFPSTWYV'

    DictAA = {AA[i]: i for i in range(len(AA))}
    DictAA1 = {AA1[i]: i for i in range(len(AA1))}

    with open(dataFile) as f:
        records = f.readlines()[1:]
    AADistance = []
    for line in records:
        array = line.strip().split()[1:]
        AADistance.append([float(x) for x in array])
    AADistance = np.array(AADistance)

    with open(dataFile1) as f:
        records = f.readlines()[1:]
    AADistance1 = []
    for line in records:
        array = line.strip().split()[1:]
        AADistance1.append([float(x) for x in array])
    AADistance1 = np.array(AADistance1)

    encodings = []
    header = []

    min_seq_length = min([len(re.sub('-', '', seq)) for seq in sequences])

    max_nlag = min(nlag, min_seq_length - 1)

    for n in range(1, max_nlag + 1):
        header.append('Schneider.lag' + str(n))
    for n in range(1, max_nlag + 1):
        header.append('Grantham.lag' + str(n))
    encodings.append(header)

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        len_seq = len(sequence)

        current_nlag = min(max_nlag, len_seq - 1)

        if current_nlag < 1:
            code = [0] * (2 * max_nlag)
            encodings.append(code)
            continue

        for n in range(1, current_nlag + 1):
            sum_distance = sum([
                (AADistance[DictAA.get(sequence[j], 0)][DictAA.get(sequence[j + n], 0)]) ** 2
                for j in range(len_seq - n)
                if sequence[j] in DictAA and sequence[j + n] in DictAA
            ])
            avg_distance = sum_distance / (len_seq - n)
            code.append(avg_distance)

        if current_nlag < max_nlag:
            code.extend([0] * (max_nlag - current_nlag))

        for n in range(1, current_nlag + 1):
            sum_distance = sum([
                (AADistance1[DictAA1.get(sequence[j], 0)][DictAA1.get(sequence[j + n], 0)]) ** 2
                for j in range(len_seq - n)
                if sequence[j] in DictAA1 and sequence[j + n] in DictAA1
            ])
            avg_distance = sum_distance / (len_seq - n)
            code.append(avg_distance)

        if current_nlag < max_nlag:
            code.extend([0] * (max_nlag - current_nlag))

        encodings.append(code)

    return encodings
