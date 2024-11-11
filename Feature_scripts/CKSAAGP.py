#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair

def get_CKSAAGP(sequences, gap=3):
    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = list(group.keys())

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []
    header = []
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p + '.gap' + str(g))
    encodings.append(header)

    for sequence in sequences:
        sequence = re.sub('-', '', sequence)
        code = []
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            total = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    key1 = index[sequence[p1]]
                    key2 = index[sequence[p2]]
                    gPair[key1 + '.' + key2] += 1
                    total += 1

            if total == 0:
                code.extend([0] * len(gPairIndex))
            else:
                code.extend([gPair[gp] / total for gp in gPairIndex])
        encodings.append(code)

    return encodings
