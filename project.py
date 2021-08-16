import sys
import numpy as np
import pandas as pd
import argparse as ap
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

parser = ap.ArgumentParser(description="Picks out an optimal mouse-human projection based on an annotation alignment")
parser.add_argument("-i", "--input_data", help="Input datafile containing the aligned annotations", nargs='?')
parser.add_argument("-f", "--full_data", help="Full bin by annotation datafile", nargs='?')
parser.add_argument("-o", "--output_prefix", help="Prefix for the output datafile, default './out'", type=str, default="out", const="out", nargs='?')
parser.add_argument("-m", "--method", help="Either use nearest neighbor (neighbor) or clustering (cluster)", default="neighbor", const="neighbor", nargs='?')
parser.add_argument("-hc", "--hcnt", help="Number of human annotations in the dataset to be aligned", type=int, default="8824", const="8824", nargs='?')
parser.add_argument("-mc", "--mcnt", help="Number of mouse annotations in the dataset to be aligned", type=int, default="3113", const="3113", nargs='?')

args = parser.parse_args()

HUMAN_CNT = args.hcnt
MOUSE_CNT = args.mcnt
K_COARSE = 20.0
K_THRESH = 0.2

m = pd.read_csv(args.input_data, sep='\t', header=0)
m = m[m.columns[:-3]]
m_human = m.iloc[:HUMAN_CNT]
m_mouse = m.iloc[HUMAN_CNT:]

full = pd.read_csv(args.full_data, sep='\t', header=None).T
n_bin = len(full.columns)

if args.method == 'neighbor':
    projection = np.zeros((2*n_bin, MOUSE_CNT))
    neighbors = np.zeros(MOUSE_CNT)
    nb_tool = NearestNeighbors(n_neighbors=MOUSE_CNT, algorithm='ball_tree').fit(m_human)
    indices = nb_tool.kneighbors(m_mouse, return_distance=False)
    for i in range(MOUSE_CNT):
        for j in range(MOUSE_CNT):
            check = indices[i][j]
            if check in neighbors:
                continue
            neighbors[i] = check
            break
    for ind in full.columns:
        projection[ind] = full[ind][neighbors]
        projection[ind + n_bin] = full[ind][HUMAN_CNT:]
    (pd.DataFrame(projection).T).to_csv(args.output_prefix + '.projection.txt', sep='\t', header=False, index=False)
    sys.exit()

k_range = [50]
inertias = []
models = []

for k in k_range:
    kmeans = KMeans(n_clusters=k).fit(m)
    this_inertia = kmeans.inertia_
    if (len(inertias)>0 and (inertias[-1] - this_inertia)/inertias[-1] < K_THRESH):
        break
    models.append(kmeans)
    inertias.append(this_inertia)

opt_k = k_range[len(models)-1]
labels = (models[-1]).labels_
h_labels = labels[:HUMAN_CNT]
m_labels = labels[HUMAN_CNT:]
projection = np.zeros((2*n_bin, opt_k))

for ind in full.columns:
    projection[ind] = [np.mean(full[ind][np.where(h_labels==i)[0]]) for i in range(opt_k)]
    projection[ind + n_bin] = [np.mean(full[ind][np.where(m_labels==i)[0]]) for i in range(opt_k)]

(pd.DataFrame(projection).T).to_csv(args.output_prefix + '.projection.txt', sep='\t', header=False, index=False)
