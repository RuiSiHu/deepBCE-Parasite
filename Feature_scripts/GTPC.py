#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re
from collections import Counter

def get_GTPC(sequences):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'positivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKeys = list(group.keys())
    triplets = [g1 + '.' + g2 + '.' + g3 for g1 in groupKeys for g2 in groupKeys for g3 in groupKeys]
    encodings = []
    header = triplets
    encodings.append(header)

    index = {}
    for key in groupKeys:
        for aa in group[key]:
            index[aa] = key

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        myDict = {tp: 0 for tp in triplets}
        total = 0

        for j in range(len(sequence) - 2):
            if sequence[j] in index and sequence[j + 1] in index and sequence[j + 2] in index:
                key = index[sequence[j]] + '.' + index[sequence[j + 1]] + '.' + index[sequence[j + 2]]
                myDict[key] += 1
                total += 1

        if total == 0:
            code = [0] * len(triplets)
        else:
            code = [myDict[tp] / total for tp in triplets]
        encodings.append(code)

    return encodings
