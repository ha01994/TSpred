import os, sys, math, glob, csv
import numpy as np
import random
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.load_dataset import Dataset
from models.model_cnn import cnnmodel
from models.model_att import attmodel
from utils.evaluate import *
from utils.other import *
import tqdm
import shutil




if os.path.isdir('example_run/losses/'):
    shutil.rmtree('example_run/losses/')
if os.path.isdir('example_run/save_dir/'):
    shutil.rmtree('example_run/save_dir/')

os.mkdir('example_run/losses/')
os.mkdir('example_run/losses/cnn')
os.mkdir('example_run/losses/att')
os.mkdir('example_run/save_dir/')
os.mkdir('example_run/save_dir/cnn')
os.mkdir('example_run/save_dir/att')




def get_loss(pred, label):
    loss_op = nn.BCELoss()
    loss = loss_op(pred, label)
    return loss





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
    for i in tqdm.tqdm(range(j)):
        batch_idx = pair_idx[i*bsz: (i+1)*bsz]
        batch_pep_idx = pairs_pep_indices[batch_idx]
        batch_tcr_idx = pairs_tcr_indices[batch_idx]
        label = torch.FloatTensor(all_label_mat[batch_idx]).to(device)
        
        ft_dict = load_ft_dict(protein_ft_dict, batch_pep_idx, batch_tcr_idx)

        if model_type=='cnn':
            pred = model(**ft_dict)
        else:
            pred,_,_,_ = model(**ft_dict)
        loss = get_loss(pred, label)
        losses.append(loss.cpu().detach().item())
        preds.extend(list(pred.cpu().detach().numpy()))
        labels.extend(list(label.cpu().detach().numpy()))
    epoch_loss = np.mean(losses)
    epoch_auroc, epoch_auprc = evaluation_metrics(preds, labels)

    return epoch_loss, epoch_auroc, epoch_auprc, preds, labels


           
    
    



save_dir = 'example_run/save_dir/cnn/'
result_dir = 'example_run/losses/cnn/'

set_random_seed()

dataset = Dataset('example_run/features/')
protein_ft_dict = dataset.to_tensor(device)
all_label_mat, pairs_pep_indices, pairs_tcr_indices,\
max_pep_len, train_index, val_index, test_index,\
max_a1_len, max_a2_len, max_a3_len, max_b1_len, max_b2_len, max_b3_len,\
= dataset.get_stuff()

print('len(train_index)', len(train_index))
print('len(val_index)', len(val_index))
print('len(test_index)', len(test_index))

model = cnnmodel(max_pep_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len, dropout_cnn).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_cnn)    
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d'%total_params)

curr_val_auroc = 0
for epoch in range(1, epochs_cnn+1):            
    print('-------- epoch %d --------'%epoch)            
    pair_idx = train_index
    random.shuffle(pair_idx)

    preds, labels, losses = [], [], []
    j = int(len(pair_idx)/bsz)
    for i in tqdm.tqdm(range(j)):
        model.train()
        optimizer.zero_grad()            
        batch_idx = pair_idx[i*bsz: (i+1)*bsz]
        batch_pep_idx = pairs_pep_indices[batch_idx]
        batch_tcr_idx = pairs_tcr_indices[batch_idx]
        label = torch.FloatTensor(all_label_mat[batch_idx]).to(device)
        ft_dict = load_ft_dict(protein_ft_dict, batch_pep_idx, batch_tcr_idx)            
        pred = model(**ft_dict)            
        loss = get_loss(pred, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().item())
        preds.extend(list(pred.cpu().detach().numpy()))
        labels.extend(list(label.cpu().detach().numpy()))

    train_loss = np.mean(losses)
    train_auroc, _ = evaluation_metrics(preds, labels)

    #==================================================================================================================#
    val_loss, val_auroc, _, _, _ = run_evaluation(val_index, model, bsz, pairs_pep_indices, pairs_tcr_indices, 
                                                  all_label_mat, protein_ft_dict, 'cnn')

    test_loss, test_auroc, _, preds, labels = run_evaluation(test_index, model, bsz, pairs_pep_indices, pairs_tcr_indices, 
                                                             all_label_mat, protein_ft_dict, 'cnn')        
    #==================================================================================================================#

    print('epoch %d'%epoch)
    print('train loss : %.4f | auroc: %.4f'%(train_loss, train_auroc))
    print('val loss   : %.4f | auroc: %.4f'%(val_loss, val_auroc))
    print('test loss  : %.4f | auroc: %.4f'%(test_loss, test_auroc))

    with open(os.path.join(result_dir, 'losses.csv'), 'a') as fw:
        fw.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(epoch, train_loss, train_auroc, val_loss, val_auroc,
                                                       test_loss, test_auroc))        

    if curr_val_auroc < val_auroc:
        print('curr_val_auroc < val_auroc')
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        curr_val_auroc = val_auroc



        
        
     
        

save_dir = 'example_run/save_dir/att/'
result_dir = 'example_run/losses/att/'

set_random_seed()

dataset = Dataset('example_run/features/')
protein_ft_dict = dataset.to_tensor(device)
all_label_mat, pairs_pep_indices, pairs_tcr_indices,\
max_pep_len, train_index, val_index, test_index,\
max_a1_len, max_a2_len, max_a3_len, max_b1_len, max_b2_len, max_b3_len,\
= dataset.get_stuff()

print('len(train_index)', len(train_index))
print('len(val_index)', len(val_index))
print('len(test_index)', len(test_index))

model = attmodel(max_pep_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len, dropout_att).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_att)    
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d'%total_params)

curr_val_auroc = 0
for epoch in range(1, epochs_att+1):
    print('-------- epoch %d --------'%epoch)
    pair_idx = train_index
    random.shuffle(pair_idx)

    preds, labels, losses = [], [], []
    j = int(len(pair_idx)/bsz)
    for i in tqdm.tqdm(range(j)):
        model.train()
        optimizer.zero_grad()            
        batch_idx = pair_idx[i*bsz: (i+1)*bsz]
        batch_pep_idx = pairs_pep_indices[batch_idx]
        batch_tcr_idx = pairs_tcr_indices[batch_idx]
        label = torch.FloatTensor(all_label_mat[batch_idx]).to(device)
        ft_dict = load_ft_dict(protein_ft_dict, batch_pep_idx, batch_tcr_idx)
        pred,_,_,_ = model(**ft_dict)            
        loss = get_loss(pred, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().item())
        preds.extend(list(pred.cpu().detach().numpy()))
        labels.extend(list(label.cpu().detach().numpy()))

    train_loss = np.mean(losses)
    train_auroc, _ = evaluation_metrics(preds, labels)

    #==================================================================================================================#
    val_loss, val_auroc, _, _, _ = run_evaluation(val_index, model, bsz, pairs_pep_indices, pairs_tcr_indices, 
                                                  all_label_mat, protein_ft_dict, 'att')

    test_loss, test_auroc, _, preds, labels = run_evaluation(test_index, model, bsz, pairs_pep_indices, pairs_tcr_indices, 
                                                             all_label_mat, protein_ft_dict, 'att')        
    #==================================================================================================================#

    print('epoch %d'%epoch)
    print('train loss : %.4f | auroc: %.4f'%(train_loss, train_auroc))
    print('val loss   : %.4f | auroc: %.4f'%(val_loss, val_auroc))
    print('test loss  : %.4f | auroc: %.4f'%(test_loss, test_auroc))

    with open(os.path.join(result_dir, 'losses.csv'), 'a') as fw:
        fw.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n'%(epoch, train_loss, train_auroc, val_loss, val_auroc,
                                                       test_loss, test_auroc))        

    if curr_val_auroc < val_auroc:
        print('curr_val_auroc < val_auroc')
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        curr_val_auroc = val_auroc



        
        
        
        
        
        
