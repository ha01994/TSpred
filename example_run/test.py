import os, sys, math, glob, csv
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
sys.path.append('../')
from utils.load_dataset import Dataset
from src.model_att import attmodel
from src.model_cnn import cnnmodel
from utils.evaluate import *
from utils.other import *
import tqdm
from collections import defaultdict
import subprocess, shutil


folder = '.'


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
           'batch_b3_seq': protein_ft_dict['b3_seq'][batch_tcr_idx],
           'batch_a_seq': protein_ft_dict['a_seq'][batch_tcr_idx],
           'batch_b_seq': protein_ft_dict['b_seq'][batch_pep_idx],
           'batch_pep_sequence': np.array(protein_ft_dict['pep_sequence'])[batch_pep_idx],    
           'batch_tcr_sequence': np.array(protein_ft_dict['tcr_sequence'])[batch_tcr_idx],}




def run_evaluation(pair_idx, model, bsz, pairs_pep_indices, pairs_tcr_indices, all_label_mat, protein_ft_dict, model_type):
    model.eval()

    preds, labels, losses = [], [], []
    pep_sequences, tcr_sequences = [], []
    pep_indices, tcr_indices = [], []
    
    j = int(len(pair_idx)/bsz) if len(pair_idx)%bsz==0 else int(len(pair_idx)/bsz)+1
    for i in range(j):
        batch_idx = pair_idx[i*bsz: (i+1)*bsz]
        batch_pep_idx = pairs_pep_indices[batch_idx]
        batch_tcr_idx = pairs_tcr_indices[batch_idx]
        label = torch.FloatTensor(all_label_mat[batch_idx]).to(device)
        ft_dict = load_ft_dict(protein_ft_dict, batch_pep_idx, batch_tcr_idx)        
        
        if model_type=='cnn':
            pred = model(ft_dict['batch_pep_onehot'],
                     ft_dict['batch_a1_onehot'],
                     ft_dict['batch_a2_onehot'],
                     ft_dict['batch_a3_onehot'],
                     ft_dict['batch_b1_onehot'],
                     ft_dict['batch_b2_onehot'],
                     ft_dict['batch_b3_onehot'])
        else:
            pep = ft_dict['batch_pep_seq']
            a = ft_dict['batch_a_seq']
            b = ft_dict['batch_b_seq']
            pred,_,_,_ = model(pep,a,b)
        
        loss = get_loss(pred, label)
        losses.append(loss.cpu().detach().item())
        preds.extend(list(pred.cpu().detach().numpy()))
        labels.extend(list(label.cpu().detach().numpy()))
        pep_sequences.extend(ft_dict['batch_pep_sequence'])
        tcr_sequences.extend(ft_dict['batch_tcr_sequence'])
        pep_indices.extend(batch_pep_idx)
        tcr_indices.extend(batch_tcr_idx)
        
    loss = np.mean(losses)
    auroc, auprc = evaluation_metrics(preds, labels)

    return loss, auroc, auprc, preds, labels, pep_sequences, tcr_sequences, pep_indices, tcr_indices





for mode in ['cnn','att','ensemble']:
    print('============= MODE: %s ============='%mode)

    set_random_seed()    
    dataset = Dataset('%s/features/'%folder)
    protein_ft_dict = dataset.to_tensor(device)
    all_label_mat, pairs_pep_indices, pairs_tcr_indices,\
    max_pep_len, train_index, val_index, test_index,\
    max_a1_len, max_a2_len, max_a3_len, max_b1_len, max_b2_len, max_b3_len,\
    = dataset.get_stuff()

    with open('../utils/hyperparams.csv', 'r') as f:
        r = csv.reader(f)
        for line in r:
            if line[0]=='bsz':bsz=int(line[1])
            if line[0]=='lr_cnn':lr_cnn=float(line[1])
            if line[0]=='lr_att':lr_att=float(line[1])
            if line[0]=='epochs_cnn':epochs_cnn=int(line[1])
            if line[0]=='epochs_att':epochs_att=int(line[1])        
            if line[0]=='dropout_att':dropout_att=float(line[1])
            if line[0]=='dropout_cnn':dropout_cnn=float(line[1])

    model_cnn_ = cnnmodel(max_pep_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len,dropout_cnn).to(device)
    model_att_ = attmodel(max_pep_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len,dropout_att).to(device)

    ####################################################################################################

    model_cnn_.load_state_dict(torch.load('%s/save_dir/cnn/model.pt'%folder, map_location=device))
    _, _, _, cnn_preds, cnn_labels, pep_sequences, tcr_sequences, pep_indices, tcr_indices = run_evaluation(
        test_index, model_cnn_, bsz, pairs_pep_indices, pairs_tcr_indices,
        all_label_mat, protein_ft_dict, 'cnn')

    model_att_.load_state_dict(torch.load('%s/save_dir/att/model.pt'%folder, map_location=device))
    _, _, _, att_preds, att_labels, pep_sequences, tcr_sequences, pep_indices, tcr_indices = run_evaluation(
        test_index, model_att_, bsz, pairs_pep_indices, pairs_tcr_indices,
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
    
    if mode=='ensemble':
        with open('%s/predictions.csv'%folder, 'w') as fw:
            fw.write('pep_id,tcr_id,pep_seq,tcr_seq,label,prediction\n')
            for q,w,e,r,t,y in zip(pep_indices, tcr_indices, pep_sequences, tcr_sequences, cnn_labels, preds):
                fw.write('pep%d,tcr%d,%s,%s,%d,%.4f\n'%(q,w,e,r,t,y))

    ####################################################################################################



