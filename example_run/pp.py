# This code is adapted from the source code of DeepAAI 
# https://github.com/enai4bio/deepaai
import os, sys
import numpy as np
import pandas as pd
import json, logging
import csv, re
import pickle
from collections import defaultdict
import torch
sys.path.append('../')
from utils.dataset_tools import get_padding_ft_dict, get_index_in_target_list
from utils.other import *
import shutil


folder = '.'


def onehot_encode(seq_list, sql_len):    
    amino_one_hot_ft_pad_dict = get_padding_ft_dict()    
    ft_mat = []
    for seq in seq_list:
        ft = []
        for idx in range(sql_len):
            if idx < len(seq):
                amino_name = seq[idx]
            else:
                amino_name = 'pad'
            amino_ft = amino_one_hot_ft_pad_dict[amino_name]            
            ft.append(amino_ft)            
        ft = np.array(ft)
        ft_mat.append(ft)
    ft_mat = np.array(ft_mat).astype(np.float32)

    return ft_mat



aa_dic = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 
          'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
          'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
          'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}



def aa_to_num(list_, max_len):
    out = []
    for seq in list_:
        out.append([aa_dic[aa] for aa in seq] + [0]*(max_len-len(seq)))
    out = np.array(out)
    return out





pep_id_to_seq = {}
pep_seq_to_id = {}
with open('%s/formatted_data/ids_pep.csv'%folder, 'r') as f:
    r = csv.reader(f)
    for line in r:
        pep_id_to_seq[line[0]] = line[1]
        pep_seq_to_id[line[1]] = line[0]

tcr_id_to_seq = {}
tcr_seq_to_id = {}
with open('%s/formatted_data/ids_tcr.csv'%folder, 'r') as f:
    r = csv.reader(f)
    for line in r:
        tcr_id_to_seq[line[0]] = line[1]
        tcr_seq_to_id[line[1]] = line[0]        
        

        


path = '%s/features/'%folder
if os.path.isdir(path):
    shutil.rmtree(path)
os.mkdir(path)

df = pd.read_csv('%s/formatted_data/data.csv'%folder)

split = df['split'].to_numpy()
seen_index = np.where(split=='train')[0]
unseen_index = np.concatenate((np.where(split=='val')[0],
                               np.where(split=='test')[0]), 0)

train_index = seen_index
val_index = np.where(split=='val')[0]
test_index = np.where(split=='test')[0]


all_label_mat = df['label'].to_numpy().astype(np.compat.long)

pep_ids_list = df['pep_id'].to_list()
tcr_ids_list = df['tcr_id'].to_list()

q = sorted(list(set([int(x[3:]) for x in pep_ids_list])))
pep_ids_set = ['pep%d'%z for z in q]
q = sorted(list(set([int(x[3:]) for x in tcr_ids_list])))
tcr_ids_set = ['tcr%d'%z for z in q]

pairs_pep_indices = get_index_in_target_list(pep_ids_list, pep_ids_set)
pairs_tcr_indices = get_index_in_target_list(tcr_ids_list, tcr_ids_set)



np.save(os.path.join(path, 'train_index'), train_index)
np.save(os.path.join(path, 'val_index'), val_index)    
np.save(os.path.join(path, 'test_index'), test_index)
np.save(os.path.join(path, 'all_label_mat'), all_label_mat)
np.save(os.path.join(path, 'pairs_pep_indices'), pairs_pep_indices)
np.save(os.path.join(path, 'pairs_tcr_indices'), pairs_tcr_indices)






pep_seqs_set = [pep_id_to_seq[x] for x in pep_ids_set]
tcr_seqs_set = [''.join(tcr_id_to_seq[x].split('_')) for x in tcr_ids_set]
a1_seqs_set = [tcr_id_to_seq[x].split('_')[0] for x in tcr_ids_set]
a2_seqs_set = [tcr_id_to_seq[x].split('_')[1] for x in tcr_ids_set]
a3_seqs_set = [tcr_id_to_seq[x].split('_')[2] for x in tcr_ids_set]
b1_seqs_set = [tcr_id_to_seq[x].split('_')[3] for x in tcr_ids_set]
b2_seqs_set = [tcr_id_to_seq[x].split('_')[4] for x in tcr_ids_set]
b3_seqs_set = [tcr_id_to_seq[x].split('_')[5] for x in tcr_ids_set]
a_seqs_set = [tcr_id_to_seq[x].split('_')[0]+tcr_id_to_seq[x].split('_')[1]+tcr_id_to_seq[x].split('_')[2] for x in tcr_ids_set]
b_seqs_set = [tcr_id_to_seq[x].split('_')[3]+tcr_id_to_seq[x].split('_')[4]+tcr_id_to_seq[x].split('_')[5] for x in tcr_ids_set]



max_pep_len = max([len(x) for x in pep_seqs_set])
max_a1_len = max([len(x) for x in a1_seqs_set])
max_a2_len = max([len(x) for x in a2_seqs_set])
max_a3_len = max([len(x) for x in a3_seqs_set])
max_b1_len = max([len(x) for x in b1_seqs_set])
max_b2_len = max([len(x) for x in b2_seqs_set])
max_b3_len = max([len(x) for x in b3_seqs_set])
print('max_pep_len', max_pep_len)
print(max_a1_len)
print(max_a2_len)
print(max_a3_len)
print(max_b1_len)
print(max_b2_len)
print(max_b3_len)
print(max_a1_len+max_a2_len+max_a3_len)
print(max_b1_len+max_b2_len+max_b3_len)



    
protein_ft_dict = {}
pep_onehot = onehot_encode(pep_seqs_set, max_pep_len)
a1_onehot = onehot_encode(a1_seqs_set, max_a1_len)
a2_onehot = onehot_encode(a2_seqs_set, max_a2_len)
a3_onehot = onehot_encode(a3_seqs_set, max_a3_len)
b1_onehot = onehot_encode(b1_seqs_set, max_b1_len)
b2_onehot = onehot_encode(b2_seqs_set, max_b2_len)
b3_onehot = onehot_encode(b3_seqs_set, max_b3_len)

protein_ft_dict['pep_onehot'] = pep_onehot
protein_ft_dict['a1_onehot'] = a1_onehot
protein_ft_dict['a2_onehot'] = a2_onehot
protein_ft_dict['a3_onehot'] = a3_onehot
protein_ft_dict['b1_onehot'] = b1_onehot
protein_ft_dict['b2_onehot'] = b2_onehot
protein_ft_dict['b3_onehot'] = b3_onehot

protein_ft_dict['pep_seq'] = aa_to_num(pep_seqs_set, max_pep_len)
protein_ft_dict['a1_seq'] = aa_to_num(a1_seqs_set, max_a1_len)
protein_ft_dict['a2_seq'] = aa_to_num(a2_seqs_set, max_a2_len)
protein_ft_dict['a3_seq'] = aa_to_num(a3_seqs_set, max_a3_len)
protein_ft_dict['b1_seq'] = aa_to_num(b1_seqs_set, max_b1_len)
protein_ft_dict['b2_seq'] = aa_to_num(b2_seqs_set, max_b2_len)
protein_ft_dict['b3_seq'] = aa_to_num(b3_seqs_set, max_b3_len)
protein_ft_dict['a_seq'] = aa_to_num(a_seqs_set, max_a1_len+max_a2_len+max_a3_len)
protein_ft_dict['b_seq'] = aa_to_num(b_seqs_set, max_b1_len+max_b2_len+max_b3_len)


protein_ft_dict['pep_sequence'] = pep_seqs_set
protein_ft_dict['tcr_sequence'] = tcr_seqs_set
protein_ft_dict['a_sequence'] = a_seqs_set
protein_ft_dict['b_sequence'] = b_seqs_set


np.save(os.path.join('%s/features'%folder,'max_pep_len'), max_pep_len)
np.save(os.path.join('%s/features'%folder,'max_a1_len'), max_a1_len)
np.save(os.path.join('%s/features'%folder,'max_a2_len'), max_a2_len)
np.save(os.path.join('%s/features'%folder,'max_a3_len'), max_a3_len)
np.save(os.path.join('%s/features'%folder,'max_b1_len'), max_b1_len)
np.save(os.path.join('%s/features'%folder,'max_b2_len'), max_b2_len)
np.save(os.path.join('%s/features'%folder,'max_b3_len'), max_b3_len)





with open(os.path.join('%s/features'%folder,'protein_ft_dict.pkl'), 'wb') as f:
    pickle.dump(protein_ft_dict, f)
print('saved to protein_ft_dict.pkl')
















        
        
