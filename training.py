#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')
from tqdm.notebook import tqdm
from functools import partial
import platform

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
from torch import nn, optim
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader, Batch

from torch_geometric.nn import DataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from training import *
from model import *
from utils import *

class EarlyStopping:
    """Stops training early if validation loss does not improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint_3.pt'):
        """
        Args:
            patience (int): Number of epochs to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path to save the checkpoint file.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        

def save_model(epoch, model, optimizer, filename):
    state = {'Epoch': epoch, 
             'State_dict': model.state_dict(), 
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)
        

def train(model, device, optimizer, train_loader, criterion, args):
    epoch_train_loss = 0
    
    for i, pro in enumerate(train_loader):
        pro, labels = pro.to(device), pro.y.to(device)  
        
        optimizer.zero_grad()
        outputs = model(pro)
#        outputs.require_grad = False
        
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs.flatten(), labels)
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    epoch_train_loss /= len(train_loader)
    return model, epoch_train_loss


def test(model, device, test_loader, criterion, args):
    model.eval()
    
    data_total = []
    pred_data_total = []
    epoch_test_loss = 0
    
    with torch.no_grad():
        for i, pro in enumerate(test_loader):
            pro, labels= pro.to(device), pro.y.to(device)
            data_total += pro.y.tolist()
            
            outputs = model(pro)
            pred_data_total += outputs.view(-1).tolist()
            
            loss = criterion(outputs.flatten(), labels)
            epoch_test_loss += loss.item()
            
    epoch_test_loss /= len(test_loader)
    return data_total, pred_data_total, epoch_test_loss


def experiment(model, train_loader, test_loader, device, args, file_name ):
    time_start = time.time()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)

    list_train_loss = []
    list_test_loss = []
    print('[Train]')
    
    early_stopping = EarlyStopping(patience = args.patience, verbose=True, delta = args.delta, 
                                   path=f'checkpoint_{file_name}.pt')

    for epoch in range(args.epoch):
        scheduler.step()
        model, train_loss = train(model, device, optimizer, train_loader, criterion, args)
        data_epoch, pred_epoch, test_loss = test(model, device, test_loader, criterion, args)
        list_train_loss.append(train_loss)
        list_test_loss.append(test_loss)
        
        if epoch > 20:
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                model.load_state_dict(torch.load(f'checkpoint_{file_name}.pt'))
                break
            
            
        time_end = time.time()
        epoch_time = time_end - time_start
        time_start = time.time()
        
        print('- Epoch: {0}, Train Loss: {1:0.6f}, Test Loss: {2:0.6f} , R2: {3:0.4f}'.
              format(epoch + 1, train_loss, test_loss, r2_score( data_epoch, pred_epoch ) ) , f' Time : {epoch_time}')

    print()
    print('[Test]')
    data_total, pred_data_total, _ = test(model, device, test_loader, criterion, args)
    print('- R2: {0:0.4f}'.format(r2_score(data_total, pred_data_total)))
    
    solution = pd.DataFrame(data_total, columns=["test"])
    answer = pd.DataFrame(pred_data_total, columns=["answer"])
    csv = pd.concat([solution, answer], axis=1)
    
    args.csv = csv
    time_end = time.time()
    time_required = time_end - time_start

    args.list_train_loss = list_train_loss
    args.list_test_loss = list_test_loss
    args.data_total = data_total
    args.pred_data_total = pred_data_total

    save_model(args.epoch, model, optimizer, f'./mymodel_{file_name}.pt')  
        
    return args
