# This code is adapted from the source code of MixTCRpred 
# https://github.com/GfellerLab/MixTCRpred/blob/main/src/models.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
import sys, math
import numpy as np




def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention





class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
    
    def forward(self, q, k, v):
        batch_size, q_len, embed_dim = q.size()
        batch_size, k_len, embed_dim = k.size()
        batch_size, v_len, embed_dim = v.size()
                
        q = self.q_proj(q)    
        q = q.reshape(batch_size, self.num_heads, q_len, self.head_dim)
        k = self.k_proj(k)    
        k = k.reshape(batch_size, self.num_heads, k_len, self.head_dim)
        v = self.v_proj(v)    
        v = v.reshape(batch_size, self.num_heads, v_len, self.head_dim)
        
            
        values, attention = scaled_dot_product(q, k, v)
        #print(values.size())
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        #print(values.size())
        values = values.reshape(batch_size, q_len, embed_dim)
        o = self.o_proj(values)
        
        
        return o, attention
    
    
    

