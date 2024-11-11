#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re
import os
import platform
import math
import numpy as np

def get_QSOrder(sequences, nlag=5, weight=0.1):

    min_seq_length = min([len(seq) for seq in sequences])
    if nlag > min_seq_length - 1:
        raise ValueError(f"The lag value nlag is out of range. It should be less than {min_seq_length}.")

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
    for aa in AA1:
        header.append('Schneider.Xr.' + aa)
    for aa in AA1:
        header.append('Grantham.Xr.' + aa)
    for n in range(1, nlag + 1):
        header.append('Schneider.Xd.' + str(n))
    for n in range(1, nlag + 1):
        header.append('Grantham.Xd.' + str(n))
    encodings.append(header)

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        arraySW = []
        arrayGM = []
        len_seq = len(sequence)
        for n in range(1, nlag + 1):
            sumSW = sum([
                (AADistance[DictAA.get(sequence[j], 0)][DictAA.get(sequence[j + n], 0)]) ** 2
                for j in range(len_seq - n)
                if sequence[j] in DictAA and sequence[j + n] in DictAA
            ])
            arraySW.append(sumSW / (len_seq - n))

            sumGM = sum([
                (AADistance1[DictAA1.get(sequence[j], 0)][DictAA1.get(sequence[j + n], 0)]) ** 2
                for j in range(len_seq - n)
                if sequence[j] in DictAA1 and sequence[j + n] in DictAA1
            ])
            arrayGM.append(sumGM / (len_seq - n))

        countAA1 = {aa: sequence.count(aa) for aa in AA1}
        sumSW = sum(arraySW)
        sumGM = sum(arrayGM)
        denominatorSW = 1 + weight * sumSW
        denominatorGM = 1 + weight * sumGM

        Xr_SW = [countAA1.get(aa, 0) / denominatorSW for aa in AA1]
        Xr_GM = [countAA1.get(aa, 0) / denominatorGM for aa in AA1]

        Xd_SW = [(weight * num) / denominatorSW for num in arraySW]
        Xd_GM = [(weight * num) / denominatorGM for num in arrayGM]

        code.extend(Xr_SW)
        code.extend(Xr_GM)
        code.extend(Xd_SW)
        code.extend(Xd_GM)
        encodings.append(code)

    return encodings
