# Source: DeepAAI (https://github.com/enai4bio/DeepAAI/blob/main/processing/hiv_cls/dataset_tools.py)
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy



amino_map_idx = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
}

def get_padding_ft_dict():
    amino_one_hot_ft_pad_dict = {}
    pad_amino_map_idx = copy.deepcopy(amino_map_idx)
    padding_num = max(pad_amino_map_idx.values()) + 1 # 0-21, padding
    amino_ft_dim = padding_num + 1
    for atom_name in pad_amino_map_idx.keys():
        ft = np.zeros(amino_ft_dim)
        ft[pad_amino_map_idx[atom_name]] = 1
        amino_one_hot_ft_pad_dict[atom_name] = ft

    pad_amino_map_idx['pad'] = padding_num
    padding_ft = np.zeros(amino_ft_dim)
    padding_ft.fill(1/amino_ft_dim)
    amino_one_hot_ft_pad_dict['pad'] = padding_ft

    return amino_one_hot_ft_pad_dict


def get_index_in_target_list(need_trans_list, target_list):
    trans_index = []
    for need_trans_str in need_trans_list:
        idx = target_list.index(need_trans_str)
        trans_index.append(idx)
    return np.array(trans_index)




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
