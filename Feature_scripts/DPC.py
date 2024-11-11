#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re

def get_DPC(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    encodings = []
    header = diPeptides
    encodings.append(header)

    AADict = {AA[i]: i for i in range(len(AA))}

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 1):
            if sequence[j] in AADict and sequence[j + 1] in AADict:
                index = AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]
                tmpCode[index] += 1
        total = sum(tmpCode)
        if total != 0:
            tmpCode = [count / total for count in tmpCode]
        code.extend(tmpCode)
        encodings.append(code)

    return encodings