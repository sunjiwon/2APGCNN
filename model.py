#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_add_pool,global_mean_pool,global_max_pool 
from torch_geometric.data import DataLoader, Data
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import torch_geometric

from sklearn.model_selection import train_test_split
import warnings

# Torch imports for required functionalities
from torch.nn import ReLU, Dropout

from training import *
from model import *
from utils import *

class GCNlayer(nn.Module):
    def __init__(
        self, 
        n_features, 
        conv_dim=64,
        concat_dim=64,
        gc_count=3,
        pool="global_mean_pool",
        act="relu",
        dropout=0.0,
    ):

        super(GCNlayer, self).__init__()
        self.n_features = n_features
        self.conv_dim = conv_dim
        self.concat_dim = concat_dim
        self.pool = getattr(torch_geometric.nn, pool)
        self.act = getattr(F, act)
        self.dropout = dropout
        
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in range(gc_count):
            if i == 0:
                conv = GCNConv( self.n_features, self.conv_dim )
                self.conv_list.append(conv)

                bn = BatchNorm1d(self.conv_dim)
                self.bn_list.append(bn)
                
            elif i == gc_count - 1:## last dimension count to concat?? 인가
                conv = GCNConv( self.conv_dim, self.concat_dim )
                self.conv_list.append(conv)

                bn = BatchNorm1d(self.concat_dim)
                self.bn_list.append(bn)
                
            else:
                conv = GCNConv( self.conv_dim, self.conv_dim )
                self.conv_list.append(conv)

                bn = BatchNorm1d(self.conv_dim)
                self.bn_list.append(bn)

        
    def forward(self, data):
        edge_index, x = data.edge_index, data.x
        
        for conv, bn in zip(self.conv_list, self.bn_list):
            x = conv(x, edge_index)
            x = self.act(x)
            x = bn(x)
        
        x = self.pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class FClayer(nn.Module):
    def __init__(self, concat_dim, pred_dim1, out_dim, dropout):
        super(FClayer, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1  
        self.out_dim = out_dim
        self.dropout = dropout

        self.fc1 = Linear(self.concat_dim, self.pred_dim1)
        self.fc2 = Linear(self.pred_dim1, self.out_dim)
    
    def forward(self, data):
        x = self.fc1(data)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
    
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = GCNlayer(
                              args.n_features, 
                              args.conv_dim, 
                              args.concat_dim, 
                              args.gc_count, 
                              args.pool, 
                              args.act, 
                              args.dropout
                              )
        
        self.fc = FClayer(
                          args.concat_dim, 
                          args.pred_dim1, 
                          args.out_dim, 
                          args.dropout
                         )
        
    def forward(self, data):
        x = self.conv1(data)
        x = self.fc(x)
        return x





