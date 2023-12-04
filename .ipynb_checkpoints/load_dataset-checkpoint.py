import os
import numpy as np
import pandas as pd
import torch
import json
import pickle
import sys




class Dataset():
    def __init__(self, path):
        self.protein_ft_save_path = os.path.join('features','protein_ft_dict.pkl')

        self.all_label_mat = np.load(os.path.join(path, 'all_label_mat.npy'))
        self.pairs_pep_indices = np.load(os.path.join(path, 'pairs_pep_indices.npy'))
        self.pairs_tcr_indices = np.load(os.path.join(path, 'pairs_tcr_indices.npy'))
        self.max_pep_len = np.load(os.path.join('features','max_pep_len.npy'))
        self.max_tcr_len = np.load(os.path.join('features','max_tcr_len.npy'))
        
        self.max_a1_len = np.load(os.path.join('features','max_a1_len.npy'))
        self.max_a2_len = np.load(os.path.join('features','max_a2_len.npy'))
        self.max_a3_len = np.load(os.path.join('features','max_a3_len.npy'))
        self.max_b1_len = np.load(os.path.join('features','max_b1_len.npy'))
        self.max_b2_len = np.load(os.path.join('features','max_b2_len.npy'))
        self.max_b3_len = np.load(os.path.join('features','max_b3_len.npy'))

        self.train_index = np.load(os.path.join(path, 'train_index.npy'))
        self.val_index = np.load(os.path.join(path, 'val_index.npy'))        
        self.test_index = np.load(os.path.join(path, 'test_index.npy'))

        with open(self.protein_ft_save_path, 'rb') as f:
            self.protein_ft_dict = pickle.load(f)


    def get_stuff(self):        
        return self.all_label_mat, self.pairs_pep_indices, self.pairs_tcr_indices,\
                self.max_pep_len, self.max_tcr_len,\
                self.train_index, self.val_index, self.test_index,\
                self.max_a1_len, self.max_a2_len, self.max_a3_len, \
                self.max_b1_len, self.max_b2_len, self.max_b3_len, 


    def to_tensor(self, device):
        for ft_name in self.protein_ft_dict:
            if '_sequence' in ft_name:
                self.protein_ft_dict[ft_name] = self.protein_ft_dict[ft_name]
            elif '_seq' in ft_name and '_sequence' not in ft_name:
                self.protein_ft_dict[ft_name] = torch.LongTensor(self.protein_ft_dict[ft_name]).to(device)
            else:
                self.protein_ft_dict[ft_name] = torch.FloatTensor(self.protein_ft_dict[ft_name]).to(device)                
                
        return self.protein_ft_dict

