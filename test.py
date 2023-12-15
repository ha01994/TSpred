import os, sys, math, glob, csv
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.load_dataset import Dataset
from models.model_att import attmodel
from models.model_cnn import cnnmodel
from utils.evaluate import *
from utils.other import *
import tqdm
from collections import defaultdict
import subprocess, shutil





def get_loss(pred, label):
    loss_op = nn.BCELoss()
    loss = loss_op(pred, label)
    return loss




device = torch.device('cuda:0')




def load_ft_dict(protein_ft_dict, batch_pep_idx, batch_tcr_idx):
    return {'batch_pep_onehot': protein_ft_dict['pep_onehot'][batch_pep_idx],
           'batch_a1_onehot': protein_ft_dict['a1_onehot'][batch_tcr_idx],
           'batch_a2_onehot': protein_ft_dict['a2_onehot'][batch_tcr_idx],
           'batch_a3_onehot': protein_ft_dict['a3_onehot'][batch_tcr_idx],
           'batch_b1_onehot': protein_ft_dict['b1_onehot'][batch_tcr_idx],
           'batch_b2_onehot': protein_ft_dict['b2_onehot'][batch_tcr_idx],
           'batch_b3_onehot': protein_ft_dict['b3_onehot'][batch_tcr_idx],
           'batch_pep_seq': protein_ft_dict['pep_seq'][batch_pep_idx],           
           'batch_a1_seq': protein_ft_dict['a1_seq'][batch_tcr_idx],
           'batch_a2_seq': protein_ft_dict['a2_seq'][batch_tcr_idx],
           'batch_a3_seq': protein_ft_dict['a3_seq'][batch_tcr_idx],
           'batch_b1_seq': protein_ft_dict['b1_seq'][batch_tcr_idx],
           'batch_b2_seq': protein_ft_dict['b2_seq'][batch_tcr_idx],
           'batch_b3_seq': protein_ft_dict['b3_seq'][batch_tcr_idx]}




def run_evaluation(pair_idx, model, bsz, pairs_pep_indices, pairs_tcr_indices, all_label_mat, protein_ft_dict, model_type):
    model.eval()

    preds, labels, losses = [], [], []
    j = int(len(pair_idx)/bsz) if len(pair_idx)%bsz==0 else int(len(pair_idx)/bsz)+1
    for i in range(j):
        batch_idx = pair_idx[i*bsz: (i+1)*bsz]
        batch_pep_idx = pairs_pep_indices[batch_idx]
        batch_tcr_idx = pairs_tcr_indices[batch_idx]
        label = torch.FloatTensor(all_label_mat[batch_idx]).to(device)
        ft_dict = load_ft_dict(protein_ft_dict, batch_pep_idx, batch_tcr_idx)        
        
        if model_type=='att':
            pred, att_pep, att_a, att_b = model(**ft_dict)
        else:
            pred = model(**ft_dict)
        
        loss = get_loss(pred, label)
        losses.append(loss.cpu().detach().item())
        preds.extend(list(pred.cpu().detach().numpy()))
        labels.extend(list(label.cpu().detach().numpy()))
        
    loss = np.mean(losses)
    auroc, auprc = evaluation_metrics(preds, labels)    

    return loss, auroc, auprc, preds, labels





for mode in ['cnn','att','ensemble']:
    print('============= MODE: %s ============='%mode)

    set_random_seed()    
    dataset = Dataset('features/')
    protein_ft_dict = dataset.to_tensor(device)
    all_label_mat, pairs_pep_indices, pairs_tcr_indices,\
    max_pep_len, max_tcr_len, train_index, val_index, test_index,\
    max_a1_len, max_a2_len, max_a3_len, max_b1_len, max_b2_len, max_b3_len,\
    = dataset.get_stuff()

    with open('utils/hyperparams.csv', 'r') as f:
        r = csv.reader(f)
        for line in r:
            if line[0]=='bsz':bsz=int(line[1])
            if line[0]=='lr_cnn':lr_cnn=float(line[1])
            if line[0]=='lr_att':lr_att=float(line[1])
            if line[0]=='epochs_cnn':epochs_cnn=int(line[1])
            if line[0]=='epochs_att':epochs_att=int(line[1])        
            if line[0]=='dropout_att':dropout_att=float(line[1])
            if line[0]=='dropout_cnn':dropout_cnn=float(line[1])

    model_cnn_ = cnnmodel(max_pep_len, max_tcr_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len,dropout_cnn).to(device)
    model_att_ = attmodel(max_pep_len, max_tcr_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len,dropout_att).to(device)

    ####################################################################################################

    model_cnn_.load_state_dict(torch.load('save_dir/cnn/model.pt', map_location=device))
    _, _, _, cnn_preds, cnn_labels = run_evaluation(test_index, model_cnn_, bsz, pairs_pep_indices, pairs_tcr_indices,
                                                    all_label_mat, protein_ft_dict, 'cnn')

    model_att_.load_state_dict(torch.load('save_dir/att/model.pt', map_location=device))
    _, _, _, att_preds, att_labels = run_evaluation(test_index, model_att_, bsz, pairs_pep_indices, pairs_tcr_indices,
                                                    all_label_mat, protein_ft_dict, 'att')

    if mode=='ensemble':
        preds = [sum(x)/2. for x in zip(cnn_preds, att_preds)]
    elif mode=='cnn':
        preds = cnn_preds
    elif mode=='att':
        preds = att_preds

    test_auroc, test_auprc = evaluation_metrics(preds, cnn_labels)
    acc,pre,spe,recall,f1 = evaluation_metrics2(preds, cnn_labels)
    
    print('ROC-AUC: %.4f'%test_auroc)
    print('PR-AUC: %.4f'%test_auprc)    

    ####################################################################################################



