#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re

def get_ASDC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    aaPairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    encodings = []
    header = aaPairs
    encodings.append(header)

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        sum_pairs = 0
        pair_dict = {pair: 0 for pair in aaPairs}

        # 统计所有可能的二肽组合
        for j in range(len(sequence)):
            for k in range(j + 1, len(sequence)):
                aa1 = sequence[j]
                aa2 = sequence[k]
                if aa1 in AA and aa2 in AA:
                    pair = aa1 + aa2
                    pair_dict[pair] += 1
                    sum_pairs += 1

        # 计算特征值
        if sum_pairs == 0:
            code = [0] * len(aaPairs)
        else:
            code = [pair_dict[pair] / sum_pairs for pair in aaPairs]

        encodings.append(code)

    return encodings
