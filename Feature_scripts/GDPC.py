#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re
from collections import Counter

def get_GDPC(sequences):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'positivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    groupKeys = list(group.keys())
    dipeptides = [g1 + '.' + g2 for g1 in groupKeys for g2 in groupKeys]
    encodings = []
    header = dipeptides
    encodings.append(header)

    index = {}
    for key in groupKeys:
        for aa in group[key]:
            index[aa] = key

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        myDict = {dp: 0 for dp in dipeptides}
        total = 0

        for j in range(len(sequence) - 1):
            if sequence[j] in index and sequence[j + 1] in index:
                key = index[sequence[j]] + '.' + index[sequence[j + 1]]
                myDict[key] += 1
                total += 1

        if total == 0:
            code = [0] * len(dipeptides)
        else:
            code = [myDict[dp] / total for dp in dipeptides]
        encodings.append(code)

    return encodings
