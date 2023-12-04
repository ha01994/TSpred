import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
import sys, math
from att_utils import *
import numpy as np




class attention_block(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        super(attention_block, self).__init__()                
        self.dropout = dropout
                
        self.self_attn1 = MultiheadAttention(input_dim, input_dim, num_heads)
        self.self_attn2 = MultiheadAttention(input_dim, input_dim, num_heads)
        self.self_attn3 = MultiheadAttention(input_dim, input_dim, num_heads)
        
        self.ffn1 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),            
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),            
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.ffn3 = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),            
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.mha1 = MultiheadAttention(input_dim, input_dim, num_heads)
        self.mha2 = MultiheadAttention(input_dim, input_dim, num_heads)
        self.mha3 = MultiheadAttention(input_dim, input_dim, num_heads)
        
        self.layernorm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, pep,a,b):
        
        pep_, _ = self.self_attn1(pep,pep,pep)
        pep = pep + self.dropout(pep_)
        pep = self.layernorm(pep)
                
        a_, _ = self.self_attn2(a,a,a)
        a = a + self.dropout(a_)
        a = self.layernorm(a)
        
        b_, _ = self.self_attn3(b,b,b)
        b = b + self.dropout(b_)
        b = self.layernorm(b)
        
        
        z = torch.cat((a,b),1)
        pep_, pep_att = self.mha1(pep,z,z)
        pep = pep + self.dropout(pep_)
        pep = self.layernorm(pep)
        
        a_, a_att = self.mha2(a,pep,pep)
        a = a + self.dropout(a_)
        a = self.layernorm(a)
        
        b_, b_att = self.mha3(b,pep,pep)
        b = b + self.dropout(b_)
        b = self.layernorm(b)   
        
        
        pep_ = self.ffn1(pep)
        pep = pep + self.dropout(pep_)
        pep = self.layernorm(pep)
        
        a_ = self.ffn2(a)
        a = a + self.dropout(a_)
        a = self.layernorm(a)
                
        b_ = self.ffn3(b)
        b = b + self.dropout(b_)
        b = self.layernorm(b)
        
        
        return pep, a, b, pep_att, a_att, b_att
    
        
    
    

class attmodel(nn.Module):
    def __init__(self, max_pep_len, max_tcr_len, max_a1_len, max_a2_len, max_a3_len, max_b1_len, max_b2_len, max_b3_len, dropout):
        super(attmodel, self).__init__()
        self.dropout = dropout
        max_pep_len, max_tcr_len = int(max_pep_len), int(max_tcr_len)
        max_a1_len, max_a2_len, max_a3_len = int(max_a1_len), int(max_a2_len), int(max_a3_len)
        max_b1_len, max_b2_len, max_b3_len = int(max_b1_len), int(max_b2_len), int(max_b3_len)
        max_a_len = max_a1_len+max_a2_len+max_a3_len
        max_b_len = max_b1_len+max_b2_len+max_b3_len
                
        attn_dim = 128
        self.attn_dim = attn_dim
        self.emb1 = nn.Embedding(21, attn_dim, padding_idx=0)
        self.emb2 = nn.Embedding(21, attn_dim, padding_idx=0)
        self.emb3 = nn.Embedding(21, attn_dim, padding_idx=0)
                        
        n_head = 4
        d_model, d_inner = attn_dim, attn_dim
        d_k, d_v = int(attn_dim/n_head), int(attn_dim/n_head)
        
        self.attention_model = attention_model(num_layers=1, 
                                              input_dim=attn_dim,
                                              dim_feedforward = attn_dim,
                                              num_heads=n_head, 
                                              dropout=dropout)
        
        self.out_mlp = nn.Sequential(
            nn.Linear(attn_dim*(max_pep_len+max_a_len+max_b_len), attn_dim),
            nn.BatchNorm1d(attn_dim),
            nn.ReLU(inplace=True),
            nn.Linear(attn_dim, 1)
        )
        
            
    def forward(self, **ft_dict):
        pep = ft_dict['batch_pep_seq']
        a = torch.cat([ft_dict['batch_a1_seq'],ft_dict['batch_a2_seq'],ft_dict['batch_a3_seq']], 1)
        b = torch.cat([ft_dict['batch_b1_seq'],ft_dict['batch_b2_seq'],ft_dict['batch_b3_seq']], 1)
        
        pep = self.emb1(pep) #[bsz, seq_len, attn_dim]        
        a = self.emb2(a)        
        b = self.emb3(b)
                
        pep, a, b, pep_att, a_att, b_att = self.attention_model(pep,a,b)        
                
        x = torch.cat([a, b, pep], 1)
        x = x.flatten(start_dim=1)
        
        pred = self.out_mlp(x) #[bsz, attn_dim]
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        
        return pred, pep_att, a_att, b_att
    
    



class attention_model(nn.Module):    
    def __init__(self, num_layers, **block_args):
        super(attention_model, self).__init__()
        self.layers = nn.ModuleList([attention_block(**block_args) for _ in range(num_layers)])
    
    def forward(self, pep, a, b):
        
        for l in self.layers:
            pep, a, b, pep_att, a_att, b_att = l(pep, a, b)
        
        return pep, a, b, pep_att, a_att, b_att
    
    
    


    