import sys
import numpy as np
import pandas as pd
import argparse as ap
import utils as utils
from sklearn.decomposition import PCA
from itertools import product
from scipy.spatial.distance import euclidean


parser = ap.ArgumentParser(description="Pullback an annotation alignment to the bin side")
parser.add_argument("-i", "--input_data", help="Input datafile containing the PCs", nargs='?')
parser.add_argument("-o", "--output_prefix", help="Prefix for the output datafile, default './out'", type=str, default="out", const="out", nargs='?')
parser.add_argument("-p", "--pcs", help="Number of PCs, default 100", type=int, default=100, const=100, nargs='?')
parser.add_argument("-n", "--nearest_neighbors", help="Perform MNN alignment, also specifies number of nearest neighbors to compute for annotations, default 5", type=int, default=5, const=5, nargs='?')
parser.add_argument("-hc", "--hcnt", help="Number of human annotations in the dataset to be aligned", type=int, default="8824", const="8824", nargs='?')
parser.add_argument("-mc", "--mcnt", help="Number of mouse annotations in the dataset to be aligned", type=int, default="3113", const="3113", nargs='?')

args = parser.parse_args()

HUMAN_CNT = args.hcnt
MOUSE_CNT = args.mcnt

# PCA first

# Read annotations
dfs = pd.read_excel("SupplementaryTable1.xls", sheet_name=None)
inds = {} # Key: feature index, Value: Description of feature
seq_type = {} # Key: feature index, Value: Sequencing tech type

for d in dfs:
    if "1a" in d:
        continue
    df = dfs[d]
    padding = 0 if "Human" in d else HUMAN_CNT
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

m = pd.read_csv(args.input_data, sep='\t', header=None).T
full_inds = np.arange(len(m))
m_human = m.iloc[full_inds[full_inds < HUMAN_CNT]]
m_mouse = m.iloc[full_inds[full_inds >= HUMAN_CNT]]

# fit PCA model

pca = PCA()
pca.fit(pd.concat([m_human, m_mouse]))
variances = pca.explained_variance_ratio_
threshold = 0.9
num_components = args.pcs
m_transformed = pca.transform(pd.concat([m_human, m_mouse]))[:, :num_components]
loading_matrix = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_))

df = pd.DataFrame(m_transformed)
df.columns = ["PC %d" % (i + 1) for i in range(num_components)]
df["Species"] = ["Human"] * len(m_human) + ["Mouse"] * len(m_mouse)
df["Sequencing type"] = [seq_type[i] for i in full_inds]
df["Description"] = [inds[i] for i in full_inds]


# begin annotation alignment

m_transformed = df[df.columns[:-3]].values

# compute distances
nnmatrix = np.zeros((HUMAN_CNT, MOUSE_CNT))
cnt = 0
for i, j in product(range(HUMAN_CNT), range(MOUSE_CNT)):
    nnmatrix[i][j] = euclidean(m_transformed[i], m_transformed[j + HUMAN_CNT])
    cnt += 1
    if cnt % 2000000 == 0:
        sys.stderr.write("%d (%.2f%%) completed\n" % (cnt, 100 * cnt / (nnmatrix.shape[0] * nnmatrix.shape[1])))

# find nearest neighbors
num_nns = args.nearest_neighbors
human_nn = np.zeros((HUMAN_CNT, MOUSE_CNT))
mouse_nn = np.zeros((HUMAN_CNT, MOUSE_CNT))
for i in range(HUMAN_CNT):
    human_nn[i, np.argsort(nnmatrix[i])[:num_nns]] = 1
for i in range(MOUSE_CNT):
    mouse_nn[np.argsort(nnmatrix[:, i])[:num_nns], i] = 1

mnns = human_nn * mouse_nn

# compute translation vector (mouse-to-human)
m_translate = np.zeros((MOUSE_CNT, m_transformed.shape[1]))
with_mnn_inds = []
for i in range(MOUSE_CNT):
    neighbors = m_transformed[[j for j in range(HUMAN_CNT) if mnns[j, i] == 1]]
    if len(neighbors) == 0:
        continue
    weights = [utils.gauss_ker(m_transformed[HUMAN_CNT + i], neighbors[j], 1) for j in range(len(neighbors))]
    m_translate[i] = np.average(neighbors - m_transformed[HUMAN_CNT + i], weights=weights, axis=0)
    with_mnn_inds.append(HUMAN_CNT + i)
with_mnn_inds = np.asarray(with_mnn_inds)
sys.stderr.write("%d annotations have MNNs\n" % len(with_mnn_inds))

p = 1
for i in range(MOUSE_CNT):
    if (i + HUMAN_CNT) in with_mnn_inds:
        continue
    current = m_transformed[HUMAN_CNT + i]
    others = m_transformed[with_mnn_inds]
    kers = [utils.gauss_ker(current, others[j], 10) for j in range(len(others))]
    m_translate[i] = np.average(m_translate[with_mnn_inds - HUMAN_CNT], weights=kers, axis=0)
	
bin_translation = pd.DataFrame([pullback(m_translate[i], loading_matrix) for i in range(MOUSE_CNT)], index = m_mouse.index.values)
m_mouse = m_mouse + bin_translation

final = pd.DataFrame(np.concatenate((m_human, m_mouse), axis=0))
final.columns = m.columns
final["Species"] = df["Species"]
final["Sequencing type"] = df["Sequencing type"]
final["Desyocription"] = df["Description"]
final.to_csv(args.output_prefix + ".pullback.txt", sep='\t', index=False)

