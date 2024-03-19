import os, sys, math, glob, csv
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
sys.path.append('../')
from src.model_att import attmodel
from src.model_cnn import cnnmodel
from utils.evaluate import *
from utils.dataset_tools import *
from utils.other import *
import tqdm
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score
import subprocess, shutil




def get_loss(pred, label):
    loss_op = nn.BCELoss()
    loss = loss_op(pred, label)
    return loss



device = torch.device('cuda:0')


pep_seqs_set = []
a1_seqs_set = []
a2_seqs_set = []
a3_seqs_set = []
b1_seqs_set = []
b2_seqs_set = []
b3_seqs_set = []
a_seqs_set = []
b_seqs_set = []

with open('test_data_sample.csv','r') as f: 
    r = csv.reader(f)
    next(r)
    for line in r:
        pep_seqs_set.append(line[0])
        a1_seqs_set.append(line[1])
        a2_seqs_set.append(line[2])
        a3_seqs_set.append(line[3])
        b1_seqs_set.append(line[4])
        b2_seqs_set.append(line[5])
        b3_seqs_set.append(line[6])
        a_seqs_set.append(line[1]+line[2]+line[3])
        b_seqs_set.append(line[4]+line[5]+line[6])
        
        


set_random_seed()

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

max_pep_len = 12
max_a1_len = 7
max_a2_len = 8
max_a3_len = 22
max_b1_len = 6
max_b2_len = 7
max_b3_len = 23
            
model_cnn_ = cnnmodel(max_pep_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len,dropout_cnn).to(device)
model_att_ = attmodel(max_pep_len, max_a1_len,max_a2_len,max_a3_len,max_b1_len,max_b2_len,max_b3_len,dropout_att).to(device)




model_cnn_.load_state_dict(torch.load('best_models/cnn_epoch191.pt', map_location=device))
model_att_.load_state_dict(torch.load('best_models/att_epoch65.pt', map_location=device))
 
cnn_preds = []
for i in range(int(len(pep_seqs_set)/bsz) +1):
    pep_onehot = onehot_encode(pep_seqs_set[bsz*i : bsz*(i+1)], max_pep_len)
    a1_onehot = onehot_encode(a1_seqs_set[bsz*i : bsz*(i+1)], max_a1_len)
    a2_onehot = onehot_encode(a2_seqs_set[bsz*i : bsz*(i+1)], max_a2_len)
    a3_onehot = onehot_encode(a3_seqs_set[bsz*i : bsz*(i+1)], max_a3_len)
    b1_onehot = onehot_encode(b1_seqs_set[bsz*i : bsz*(i+1)], max_b1_len)
    b2_onehot = onehot_encode(b2_seqs_set[bsz*i : bsz*(i+1)], max_b2_len)
    b3_onehot = onehot_encode(b3_seqs_set[bsz*i : bsz*(i+1)], max_b3_len)
    
    pred = model_cnn_(torch.FloatTensor(pep_onehot).to(device), 
                      torch.FloatTensor(a1_onehot).to(device),
                      torch.FloatTensor(a2_onehot).to(device),
                      torch.FloatTensor(a3_onehot).to(device),
                      torch.FloatTensor(b1_onehot).to(device),
                      torch.FloatTensor(b2_onehot).to(device),
                      torch.FloatTensor(b3_onehot).to(device),)
    cnn_preds.extend(list(pred.cpu().detach().numpy()))
    
att_preds = []
for i in range(int(len(pep_seqs_set)/bsz) +1):    
    pep_seq = aa_to_num(pep_seqs_set[bsz*i : bsz*(i+1)], max_pep_len)
    a_seq = aa_to_num(a_seqs_set[bsz*i : bsz*(i+1)], max_a1_len + max_a2_len + max_a3_len)
    b_seq = aa_to_num(b_seqs_set[bsz*i : bsz*(i+1)], max_b1_len + max_b2_len + max_b3_len)
    
    pred, _, _, _ = model_att_(torch.LongTensor(pep_seq).to(device),
                               torch.LongTensor(a_seq).to(device),
                               torch.LongTensor(b_seq).to(device),)
    att_preds.extend(list(pred.cpu().detach().numpy()))

assert len(cnn_preds) == len(pep_seqs_set)
assert len(att_preds) == len(pep_seqs_set)

    
    
    
preds = [sum(x)/2. for x in zip(cnn_preds, att_preds)]

with open('ensemble_predictions.csv', 'w') as fw:
    fw.write('peptide,A1,A2,A3,B1,B2,B3,prediction\n')
    for a,b,c,d,e,f,g,h in zip(pep_seqs_set,
                                a1_seqs_set, 
                                a2_seqs_set,
                                a3_seqs_set,
                                b1_seqs_set, 
                                b2_seqs_set,
                                b3_seqs_set,
                                preds):
        fw.write('%s,%s,%s,%s,%s,%s,%s,%.4f\n'%(a,b,c,d,e,f,g,h))

        
        
        
        