import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
import sys
import numpy as np




class cnnmodule1(nn.Module):
    def __init__(self, in_channel, out_channel, out_dim, kernel_size, padding, length, dropout):
        super(cnnmodule1, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=1)    
        self.out_linear = nn.Linear(length*out_channel, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x): #x - [bsz, seqlen, h_dim]
        bsz = x.size()[0]         
        x = x.transpose(1, 2) #[bsz, h_dim, seqlen]        
        x = self.conv(x) #[bsz, out_channel, seqlen+1]
        x = self.dropout(x)
        x = self.pool(x) #[bsz, out_channel, seqlen]
        x = x.view(bsz, -1) #[bsz, out_channel*seqlen]        
        x = self.out_linear(x) #[bsz, out_dim]                
        return x

    
        
    
    
    

class cnnmodel(nn.Module):
    def __init__(self, max_pep_len, max_a1_len, max_a2_len, max_a3_len, max_b1_len, max_b2_len, max_b3_len, dropout):
        super(cnnmodel, self).__init__()
        self.dropout = dropout
        self.activation = nn.ReLU()
        for m in self.modules():
            self.weights_init(m)
        max_pep_len = int(max_pep_len)
        max_a1_len, max_a2_len, max_a3_len = int(max_a1_len), int(max_a2_len), int(max_a3_len)
        max_b1_len, max_b2_len, max_b3_len = int(max_b1_len), int(max_b2_len), int(max_b3_len)        

        out_chan = 32
        out_dim = 64
        self.out_dim = out_dim
        self.cnn_pep_1 = cnnmodule1(in_channel=21, out_channel=out_chan, out_dim=out_dim, kernel_size=2, padding=1, length=max_pep_len, dropout=self.dropout)
        self.cnn_a1_1 = cnnmodule1(in_channel=21, out_channel=out_chan, out_dim=out_dim, kernel_size=2, padding=1, length=max_a1_len, dropout=self.dropout)
        self.cnn_a2_1 = cnnmodule1(in_channel=21, out_channel=out_chan, out_dim=out_dim, kernel_size=2, padding=1, length=max_a2_len, dropout=self.dropout)
        self.cnn_a3_1 = cnnmodule1(in_channel=21, out_channel=out_chan, out_dim=out_dim, kernel_size=2, padding=1, length=max_a3_len, dropout=self.dropout)
        self.cnn_b1_1 = cnnmodule1(in_channel=21, out_channel=out_chan, out_dim=out_dim, kernel_size=2, padding=1, length=max_b1_len, dropout=self.dropout)
        self.cnn_b2_1 = cnnmodule1(in_channel=21, out_channel=out_chan, out_dim=out_dim, kernel_size=2, padding=1, length=max_b2_len, dropout=self.dropout)
        self.cnn_b3_1 = cnnmodule1(in_channel=21, out_channel=out_chan, out_dim=out_dim, kernel_size=2, padding=1, length=max_b3_len, dropout=self.dropout)
        self.local_linear1 = nn.Linear(out_dim*7, out_dim*2)
        self.local_linear2 = nn.Linear(out_dim*2, out_dim)
        
        self.pred_linear = nn.Linear(out_dim, 1)
        
        

        
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)

            

    def forward(self, **ft_dict):
        bsz = ft_dict['batch_pep_onehot'].size()[0]

        pep1 = self.cnn_pep_1(ft_dict['batch_pep_onehot']).view(bsz, -1)
        a11 = self.cnn_a1_1(ft_dict['batch_a1_onehot']).view(bsz, -1)
        a21 = self.cnn_a2_1(ft_dict['batch_a2_onehot']).view(bsz, -1)
        a31 = self.cnn_a3_1(ft_dict['batch_a3_onehot']).view(bsz, -1)
        b11 = self.cnn_b1_1(ft_dict['batch_b1_onehot']).view(bsz, -1)
        b21 = self.cnn_b2_1(ft_dict['batch_b2_onehot']).view(bsz, -1)
        b31 = self.cnn_b3_1(ft_dict['batch_b3_onehot']).view(bsz, -1)
        x = torch.cat([pep1, a11, a21, a31, b11, b21, b31], dim=-1)
        x = self.activation(x)
        
        x = self.local_linear1(x) #[bsz, out_dim]
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training) 
        
        x = self.local_linear2(x) #[bsz, out_dim]
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
                       
        pred = self.pred_linear(x)
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        
        return pred

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    