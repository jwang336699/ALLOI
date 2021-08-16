import matplotlib
matplotlib.use('Agg')

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.manifold import TSNE

aligned = pd.read_csv('results/'+sys.argv[1]+'_'+sys.argv[2]+'_bin.mnn.txt', sep='\t')
aligned_vals = aligned[aligned.columns[:-3]]
hmmmap = pd.read_csv('data/mouse_human_chromhmm_map.txt', sep='\t', header=None)
hmmmap.index = list(map(lambda x: 'E'+str(x), hmmmap[hmmmap.columns[0]]))
hmmmap = hmmmap[hmmmap.columns[1]]
hmmmap = hmmmap.append(pd.Series(['?'], ['?']))

mouse = np.load('data/'+sys.argv[1]+'_'+sys.argv[2]+'_mouse_chromhmm.npy', allow_pickle=True)
human = np.load('data/'+sys.argv[1]+'_'+sys.argv[2]+'_human_chromhmm.npy', allow_pickle=True)
mouse = mouse[:(len(aligned)/2)]
human = human[:(len(aligned)/2)]
mouse[np.where(np.array(map(lambda x: len(x), mouse)) == 0)[0]] = ['????']
human[np.where(np.array(map(lambda x: len(x), human)) == 0)[0]] = ['????']
mouse = np.array(map(lambda x: x[0], mouse))
human = np.array(map(lambda x: x[0], human))
ham = hmmmap[mouse]
aligned["Raw"] = np.concatenate((human, ham))
aligned["State"] = list(map(lambda x: x.translate(None, '0123456789'), aligned["Raw"]))

cdict = {'GapArtf': '#fff5ee',
         'Quies': '#ffffff',
         'ReprPC': '#c0c0c0',
         'TxEx': '#3cb372',
         'Tx': '#006400',
         'TxWk': '#228b22',
         'PromF': '#ff4500',
         'Acet': '#fffacd',
         'TxEnh': '#acff2f',
         'BivProm': '#800080',
         'HET': '#b19cd9',
         'EnhWk': '#ffff00',
         'EnhA': '#ffa600',
         'znf': '#7fffd4',
         'DNase': '#fff34f',
         'TSS': '#ff0000'}

tsne = TSNE(n_components=2, random_state=17).fit_transform(aligned_vals)
aligned["TSNE1"] = tsne[:,0]
aligned["TSNE2"] = tsne[:,1]
species = np.concatenate((["Human"]*(len(aligned)/2), ["Mouse"]*(len(aligned)/2)))
aligned["Species"] = species

print(sys.argv[1]+' checkpoint 1')

plt.figure(figsize=(10, 10))
sns.scatterplot(x="TSNE1", y="TSNE2", data=aligned, hue="State", style="Species", alpha=0.8, palette=cdict)
plt.savefig("results/plots/"+sys.argv[1]+"_"+sys.argv[2]+"_states.png", bbox_inches='tight')

print(sys.argv[1]+' checkpoint 2')

nn = np.load('results/'+sys.argv[1]+'_'+sys.argv[2]+'_hnbrm.npy', allow_pickle=True)
nn = np.array(map(lambda x: x[0], nn))
hcorrm = {}
for ms in set(mouse):
    subset = np.where(mouse == ms)[0]
    neighbors = nn[subset]
    hcorrm[ms] = dict(Counter(aligned["Raw"].iloc[neighbors])).items()

print(sys.argv[1]+' checkpoint 3')

with open('results/'+sys.argv[1]+'_'+sys.argv[2]+'_hmmdists.json', 'w') as file:
    json.dump(hcorrm, file)

