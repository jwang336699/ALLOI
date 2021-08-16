import sys
import pandas as pd
import numpy as np
import utils as utils

coordinates = np.load('data/all_coordinates.npy')
human = np.load('data/np_human_chromhmm.npy', allow_pickle=True)
mouse = np.load('data/np_mouse_chromhmm.npy', allow_pickle=True)

h_dict = {}
for chr in set(human[:,0]):
    chr_sub = human[human[:,0] == chr]
    h_dict[chr] = chr_sub[chr_sub[:,1].argsort()]

m_dict = {}
for chr in set(mouse[:,0]):
    chr_sub = mouse[mouse[:,0] == chr]
    m_dict[chr] = chr_sub[chr_sub[:,1].argsort()]

print('starting '+sys.argv[1])

h_states, m_states = utils.label_states(coordinates[coordinates[:,0] == sys.argv[1]], h_dict, m_dict, int(sys.argv[2]), int(sys.argv[3]))

print('done '+sys.argv[1])

np.save('data/'+sys.argv[1]+'_'+sys.argv[3]+'_human_chromhmm.npy', h_states)
np.save('data/'+sys.argv[1]+'_'+sys.argv[3]+'_mouse_chromhmm.npy', m_states)

