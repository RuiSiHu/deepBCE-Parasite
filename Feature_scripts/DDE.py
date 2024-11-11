#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re
import math

def get_DDE(sequences):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    myCodons = {
        'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2,
        'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
        'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6,
        'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
    }

    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    encodings = []
    header = diPeptides
    encodings.append(header)

    myTM = []
    for pair in diPeptides:
        myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

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

        myTV = []
        length = len(sequence)
        for tm in myTM:
            myTV.append(tm * (1 - tm) / (length - 1) if length > 1 else 0)

        # 计算标准化的特征值
        for i in range(len(tmpCode)):
            if myTV[i] == 0:
                code.append(0)
            else:
                code.append((tmpCode[i] - myTM[i]) / math.sqrt(myTV[i]))

        encodings.append(code)

    return encodings
