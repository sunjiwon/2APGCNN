#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import torch
from torch import nn, optim
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool 
from torch_geometric.data import DataLoader, Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Polypeptide import three_to_one
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
from tqdm.notebook import tqdm
import time
import warnings
warnings.filterwarnings(action='ignore')



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
def aa_features(x):
    return np.array(one_of_k_encoding(x, AA))

def adjacency2edgeindex(adjacency):
    start = []
    end = []
    adjacency = adjacency - np.eye(adjacency.shape[0], dtype=int)
    for x in range(adjacency.shape[1]):
        for y in range(adjacency.shape[0]):
            if adjacency[x, y] == 1:
                start.append(x)
                end.append(y)

    edge_index = np.asarray([start, end])
    return edge_index


AMINOS =  ['CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 
           'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET']

def filter_20_amino_acids(array):
    return ( np.in1d(array.res_name, AMINOS) & (array.res_id != -1) )

def protein_analysis(pdb_id):
    pdb_name=os.listdir(pdb_path)
    protein_name=[re.sub('.pdb', '', i) for i in pdb_name]
    
    if pdb_id not in protein_name:
        file_name = rcsb.fetch(pdb_id, "pdb", pdb_path)
        array = strucio.load_structure(file_name)
        protein_mask = filter_20_amino_acids(array)
        try:
            array = array[protein_mask]
        except:
            array = array[0]
            array = array[protein_mask]
        try:
            ca = array[array.atom_name == "CA"]
        except:
            array = array[0]
            ca = array[array.atom_name == "CA"]
        seq = ''.join([three_to_one(str(i).split(' CA')[0][-3:]) for i in ca])
        threshold = 7
        cell_list = struc.CellList(ca, cell_size=threshold)
        A = cell_list.create_adjacency_matrix(threshold)
        A = np.where(A == True, 1, A)
        return [aa_features(aa) for aa in seq], adjacency2edgeindex(A)
    
    if pdb_id in protein_name:
        array = strucio.load_structure('./Data/pdb/'+pdb_id+".pdb")
        protein_mask = filter_20_amino_acids(array)
        try:
            array = array[protein_mask]
        except:
            array = array[0]
            array = array[protein_mask]
        try:
            ca = array[array.atom_name == "CA"]
        except:
            array = array[0]
            ca = array[array.atom_name == "CA"]
        seq = ''.join([three_to_one(str(i).split(' CA')[0][-3:]) for i in ca])
        threshold = 7
        cell_list = struc.CellList(ca, cell_size=threshold)
        A = cell_list.create_adjacency_matrix(threshold)
        A = np.where(A == True, 1, A)
        return [aa_features(aa) for aa in seq], adjacency2edgeindex(A)

def generate_graph(pdb, graph_path):
    done = 0
    while done == 0:
        graph_dirs = list(set([d[:-6] for d in os.listdir(graph_path)]))
        if pdb not in graph_dirs:
            try:
                save_graph(graph_path,pdb)
                done = 1
                return 1
            except:
                done = 1
                return 0
        else:
            done = 1
            return 1

def pro2vec(pdb_id):
    node_f, edge_index = protein_analysis(pdb_id)
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))
    return data


def make_pro(df, target):
    pro_key = []
    pro_value = []
    for i in range(df.shape[0]):
        pro_key.append(df['PDB'].iloc[i])
        pro_value.append(df[target].iloc[i])
    return pro_key, pro_value


    
def save_graph(graph_path, pdb_id):
    vec = pro2vec(pdb_id)
    np.save(graph_path+pdb_id+'_e.npy', vec.edge_index)
    np.save(graph_path+pdb_id+'_n.npy', vec.x)
    
def load_graph(graph_path, pdb_id):
    n = np.load(graph_path+str(pdb_id)+'_n.npy')
    e = np.load(graph_path+str(pdb_id)+'_e.npy')
    N = torch.tensor(n, dtype=torch.float)
    E = torch.tensor(e, dtype=torch.long)
    data = Data(x=N, edge_index=E)
    return data

    
def make_vec(pro_list, value_list, graph_path):
    #print(pro_list)
    dataset = MyDataset(pro_list, value_list, graph_path)
    return dataset


def make_vec_before(pro, value, graph_path):
    X = []
    Y = []
    for i in range(len(pro)):
        m = pro[i]
        y = value[i]
        v = load_graph(graph_path, m)
        if v.x.shape[0] < 100000:
            X.append(v)
            Y.append(y)
            
    for i, data in enumerate(X):
        y = Y[i]
        #data.y = torch.tensor([y], dtype=torch.long)
        data.y = torch.tensor([y], dtype=torch.float)#flaot
    return X



class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, pro_list, value_list, graph_path):
        self.pro_list = pro_list
        self.value_list = value_list
        self.graph_path = graph_path
        self.length = len(pro_list)  # Set the length to the number of samples in your dataset

    def __iter__(self):
        for pro, value in zip(self.pro_list, self.value_list):
            v = load_graph(self.graph_path, pro)
            v.y = torch.tensor([value], dtype=torch.float)
            yield v
            
    def __len__(self):
        return self.length



def custom_collate(batch):
    # Implement your custom collation logic here
    # This function should take a list of batch elements and return a batch tensor
    
    # Example implementation for concatenating the graphs in the batch:
    batched_graph = Batch.from_data_list(batch)
    return batched_graph





