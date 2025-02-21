#!/usr/bin/env python
# _*_coding:utf-8_*_
# @Author : Rui-Si Hu

import re
from collections import Counter

def get_AAC(sequences, **kw):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = list(AA)
    encodings.append(header)

    for seq in sequences:
        sequence = re.sub('-', '', seq)
        count = Counter(sequence)
        total = len(sequence)
        code = []
        for aa in AA:
            aa_count = count.get(aa, 0)
            code.append(aa_count / total if total > 0 else 0)
        encodings.append(code)
    return encodings