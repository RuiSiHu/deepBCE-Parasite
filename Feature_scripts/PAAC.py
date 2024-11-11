#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re
import os
import platform
import math

def get_PAAC(sequences, lambdaValue=5, weight=0.05):
    dataFile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'PAAC.txt')
    if not os.path.exists(dataFile):
        raise FileNotFoundError(f"Data file not found: {dataFile}")

    with open(dataFile) as f:
        records = f.readlines()

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {AA[i]: i for i in range(len(AA))}

    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split()
        if array:
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])

    # 标准化属性值
    AAProperty1 = []
    for prop in AAProperty:
        meanProp = sum(prop) / 20
        stdProp = math.sqrt(sum([(p - meanProp) ** 2 for p in prop]) / 20)
        AAProperty1.append([(p - meanProp) / stdProp for p in prop])

    encodings = []
    header = [f"Xc1.{aa}" for aa in AA] + [f"Xc2.lambda{n}" for n in range(1, lambdaValue + 1)]
    encodings.append(header)

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        if len(sequence) < lambdaValue + 1:
            raise ValueError(f"Sequence length must be larger than lambdaValue + 1: {lambdaValue + 1}")
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            theta_n = sum([
                Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1)
                for j in range(len(sequence) - n)
            ]) / (len(sequence) - n)
            theta.append(theta_n)

        # 计算 Xc1
        AA_count = {aa: sequence.count(aa) for aa in AA}
        total_count = len(sequence)
        Xc1 = [AA_count[aa] / (1 + weight * sum(theta)) for aa in AA]

        # 计算 Xc2
        Xc2 = [(weight * t) / (1 + weight * sum(theta)) for t in theta]

        code = Xc1 + Xc2
        encodings.append(code)

    return encodings

def Rvalue(aa1, aa2, AADict, AAProperty1):
    return sum([
        AAProperty1[prop_index][AADict[aa1]] * AAProperty1[prop_index][AADict[aa2]]
        for prop_index in range(len(AAProperty1))
    ]) / len(AAProperty1)
