#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import argparse as ap
from sklearn.decomposition import PCA
from itertools import product
from scipy.spatial.distance import euclidean
from utils import encoder

parser = ap.ArgumentParser(description="Perform PCA on a matrix of positions by epigenetic annotations")
parser.add_argument("-i", "--input_data", help="An input tab-delimited datafile, with each row being a position and each column being an epigenome", nargs='?')
parser.add_argument("-o", "--output_prefix", help="Prefix for the output datafile, default './out'", type=str, default="out", const="out", nargs='?')
parser.add_argument("-p", "--pcs", help="Number of PCs, default 100", type=int, default=100, const=100, nargs='?')
parser.add_argument("-v", "--var", help="Variance threshold for number of PCs", type=float, default=0.9, const=0.9, nargs='?')
parser.add_argument("-b", "--bins", help="Whether the input points are bins", type=bool, default=False, const=False, nargs='?')


args = parser.parse_args()

# Read annotations
dfs = pd.read_excel("SupplementaryTable1.xls", sheet_name=None)
inds = {} # Key: feature index, Value: Description of feature
seq_type = {} # Key: feature index, Value: Sequencing tech type

for d in dfs:
    if "1a" in d:
        continue
    df = dfs[d]
    padding = 0 if "Human" in d else 8824
    if "feature index start" in df.columns:
        starts = list(df["feature index start"])
        ends = list(df["feature index end"])
        filenames = list(df["filename"])
        num_states = ends[0] + 1 - starts[0]
        for i in range(len(starts)):
            for j in range(num_states):
                inds[starts[i] + j + padding] = filenames[i] + "_%d" % (j + 1)
                seq_type[starts[i] + j + padding] = d.split(' ')[-1]
    else:
        indices = list(df["feature index"])
        for i in range(len(indices)):
            inds[indices[i] + padding] = ','.join([encoder(item) for item in list(df.iloc[i][df.columns[1:]])])
            seq_type[indices[i] + padding] = d.split(' ')[-1]


# Read matrix
m = pd.read_csv(args.input_data, sep='\t', header=None).T

# Remove zeroes
# full_inds = np.arange(len(m))[np.sum(m, axis=1) != 0]
full_inds = np.arange(len(m))
m_human = m.iloc[full_inds[full_inds < 8824]]
m_mouse = m.iloc[full_inds[full_inds >= 8824]]

# PCA
pca = PCA()
pca.fit(pd.concat([m_human, m_mouse]))
variances = pca.explained_variance_ratio_
threshold = args.var
num_components = args.pcs if (args.pcs > 0) else np.where(np.cumsum(pca.explained_variance_ratio_)>threshold)[0][0]
m_transformed = pca.transform(pd.concat([m_human, m_mouse]))[:, :num_components]

# Write to file
df = pd.DataFrame(m_transformed)
df.columns = ["PC %d" % (i + 1) for i in range(num_components)]
df["Species"] = ["Human"] * len(m_human) + ["Mouse"] * len(m_mouse) if not(args.bins) else 0
df["Sequencing type"] = [seq_type[i] for i in full_inds] if not(args.bins) else 0
df["Description"] = [inds[i] for i in full_inds] if not(args.bins) else 0
df.to_csv(args.output_prefix + ".pca.txt", sep='\t', index=False)
