#!/usr/bin/env python
# _*_coding:utf-8_*_
import re

def read_protein_sequences(file):
    with open(file) as f:
        data = f.read()
    if re.search('>', data) == None:
        raise ValueError("Please input correct FASTA format protein sequence！！！")
    else:
        records = data.split('>')[1:]
        sequences = []
        sequence_name = []
        for fasta in records:
            array = fasta.split('\n')
            header, sequence = array[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
            name = header
            sequences.append(sequence)
            sequence_name.append(str(name))
        return sequences, sequence_name