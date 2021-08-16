#!/usr/bin/env python

import sys, random
import numpy as np
import pandas as pd

H_FEATURES = 8824
M_FEATURES = 3113
sys.stdout = open(sys.argv[1], "w+")
datafile1 = sys.argv[2]
datafile2 = sys.argv[3]
bin_size = int(sys.argv[4])
keep_chr = sys.argv[5]
n_features = H_FEATURES + M_FEATURES

def process_line(line, human=True):
    num_features = H_FEATURES if human else M_FEATURES
    sections = line.split('|')
    pos = sections[0].split()
    features = np.asarray(sections[1].split(), dtype=int)
    data = np.asarray(sections[2].split(), dtype=float)
    output = np.zeros(num_features, dtype=float)
    # Last features are float
    if len(data) != 0:
        output[features[-1 * len(data):]] = data
    # First features are binary
    if len(features) - len(data) != 0:
        output[features[:len(features) - len(data)]] = 1
    return(output,pos[0]==keep_chr)

with open(datafile1) as file1, open(datafile2) as file2:
    line1 = file1.readline()
    line2 = file2.readline()
    os = np.zeros(n_features, dtype=float)
    cnt = 0
    while line1 and line2:
        # sys.stderr.write(line1)
        h_line,match = process_line(line1, True)
        m_line,_ = process_line(line2, False)
        if not(match):
            line1 = file1.readline()
            line2 = file2.readline()
            continue
        os += np.concatenate((h_line, m_line))
        cnt += 1
        line1 = file1.readline()
        line2 = file2.readline()
        if cnt == bin_size or not(line1 and line2):
            output = os / cnt
            sys.stdout.write('\t'.join([str(i) for i in output]) + '\n')
            os = np.zeros(n_features, dtype=float)
            cnt = 0
