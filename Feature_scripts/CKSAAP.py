#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re

def get_CKSAAP(sequences, gap=3):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    aaPairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]

    encodings = []
    header = []
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        for g in range(gap + 1):
            myDict = {pair: 0 for pair in aaPairs}
            total = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] += 1
                    total += 1
            if total == 0:
                code.extend([0] * len(aaPairs))
            else:
                code.extend([myDict[pair] / total for pair in aaPairs])
        encodings.append(code)

    return encodings
