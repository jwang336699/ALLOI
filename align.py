#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import argparse as ap
import utils as utils
from sklearn.decomposition import PCA
from itertools import product
from scipy.spatial.distance import euclidean


parser = ap.ArgumentParser(description="Perform human-mouse alignment on the PCA embedded data")
parser.add_argument("-i", "--input_data", help="Input datafile containing the PCs", nargs='?')
parser.add_argument("-o", "--output_prefix", help="Prefix for the output datafile, default './out'", type=str, default="out", const="out", nargs='?')
parser.add_argument("-n", "--nearest_neighbors", help="Perform MNN alignment, also specifies number of nearest neighbors to compute for annotations, default 5", type=int, default=5, const=5, nargs='?')
parser.add_argument("-m", "--mode", help="One of ""average"", ""neighbors"" and ""smooth"", only used in conjuction with -n argument", type=str, default="neighbors", const="neighbors", nargs='?')
parser.add_argument("-a", "--harmony", help="Perform Harmony alignment", action="store_true")
parser.add_argument("-hc", "--hcnt", help="Number of human annotations in the dataset to be aligned", type=int, default="8824", const="8824", nargs='?')
parser.add_argument("-mc", "--mcnt", help="Number of mouse annotations in the dataset to be aligned", type=int, default="3113", const="3113", nargs='?')
parser.add_argument("-b", "--bin", help="Whether the current alignment is of bins", type=bool, default=False, const=False, nargs='?')
parser.add_argument("-s", "--sigma", help="Sigma for the Gaussian smoothing", type=float, default="1", const="1", nargs='?')
parser.add_argument("-sn", "--sigman", help="Sigma for the Gaussian smoothing (MNN-less bins)", type=float, default="30", const="30", nargs='?')

args = parser.parse_args()

HUMAN_CNT = args.hcnt
MOUSE_CNT = args.mcnt

# Read input PCs
m = pd.read_csv(args.input_data, sep='\t')
m_transformed = m[m.columns[:-3]].values

if args.bin:
    HUMAN_CNT = len(m)/2
    MOUSE_CNT = len(m)/2

if args.harmony:
    ho = hm.run_harmony(data_mat=m_transformed, meta_data=m, vars_use=["Species"], max_iter_harmony=50)
    df_harmony = pd.DataFrame(ho.Z_corr).T
    df_harmony.columns = ['X{}'.format(i + 1) for i in range(df_harmony.shape[1])]
    df_harmony["Species"] = m["Species"]
    df_harmony["Sequencing type"] = m["Sequencing type"]
    df_harmony["Description"] = m["Description"]
    df_harmony.to_csv(args.output_prefix + ".harmony.txt", sep='\t', index=False)
    
else:
    # Compute distances
    nnmatrix = np.zeros((HUMAN_CNT, MOUSE_CNT))
    cnt = 0
    for i, j in product(range(HUMAN_CNT), range(MOUSE_CNT)):
        nnmatrix[i][j] = euclidean(m_transformed[i], m_transformed[j + HUMAN_CNT])
        cnt += 1
        if cnt % 2000000 == 0:
            sys.stderr.write("%d (%.2f%%) completed\n" % (cnt, 100 * cnt / (nnmatrix.shape[0] * nnmatrix.shape[1])))

    # Find nearest neighbors of each point
    num_nns = args.nearest_neighbors
    human_nn = np.zeros((HUMAN_CNT, MOUSE_CNT))
    mouse_nn = np.zeros((HUMAN_CNT, MOUSE_CNT))
    for i in range(HUMAN_CNT):
        human_nn[i, np.argsort(nnmatrix[i])[:num_nns]] = 1
    for i in range(MOUSE_CNT):
        mouse_nn[np.argsort(nnmatrix[:, i])[:num_nns], i] = 1

    # Find mutual nearest neighbors
    mnns = human_nn * mouse_nn

    # Set human as reference. For each mouse epigenome, compute average translation vector
    m_translate = np.zeros((MOUSE_CNT, m_transformed.shape[1]))
    with_mnn_inds = []
    for i in range(MOUSE_CNT):
        neighbors = m_transformed[[j for j in range(HUMAN_CNT) if mnns[j, i] == 1]]
        if len(neighbors) == 0:
            continue
        weights = [utils.gauss_ker(neighbors[j], m_transformed[HUMAN_CNT + i], args.sigma) for j in range(len(neighbors))]
        m_translate[i] = np.average(neighbors - m_transformed[HUMAN_CNT + i], weights=weights, axis=0)
        # m_translate[i] = np.mean(neighbors - m_transformed[HUMAN_CNT + i], axis=0)
        with_mnn_inds.append(HUMAN_CNT + i)
    with_mnn_inds = np.asarray(with_mnn_inds)
    sys.stderr.write("%d annotations have MNNs\n" % len(with_mnn_inds))

    m_mouse_translated = m_transformed[HUMAN_CNT:]
    if args.mode == "average":
        # Global average translation vector
        m_translate_avg = np.mean(m_translate[np.sum(m_translate, axis=1) != 0], axis=0)
        # Translate mouse
        m_mouse_translated = m_transformed[HUMAN_CNT:] + m_translate_avg

    elif args.mode == "neighbors":
        p = 1
        # If a point does not have mutual nearest neighbors,
        # its translation vector of each point is a distance-weighted sum of
        # the average translation vector of all points with mutual nearest neighbors
        for i in range(MOUSE_CNT):
            if (i + HUMAN_CNT) in with_mnn_inds:
                continue
            current = m_transformed[HUMAN_CNT + i]
            others = m_transformed[with_mnn_inds]
            kers = [utils.gauss_ker(current, others[j], args.sigman) for j in range(len(others))]
            m_translate[i] = np.average(m_translate[with_mnn_inds - HUMAN_CNT], weights=kers, axis=0)
            # dists = np.sqrt(np.sum((others - current) ** 2, axis=1))
            # weights = 1 / (dists ** p)
            # weights_normed = weights / np.sum(weights)
            # m_translate[i] = np.sum(m_translate[with_mnn_inds - HUMAN_CNT] * (weights_normed.reshape(-1, 1)), axis=0)
        m_mouse_translated = m_transformed[HUMAN_CNT:] + m_translate

    m_human_translated = m_transformed[:HUMAN_CNT]

    # Write to file
    df = pd.DataFrame(np.concatenate((m_human_translated, m_mouse_translated), axis=0))
    df.columns = m.columns[:-3]
    df["Species"] = m["Species"]
    df["Sequencing type"] = m["Sequencing type"]
    df["Description"] = m["Description"]
    df.to_csv(args.output_prefix + ".mnn.txt", sep='\t', index=False)
