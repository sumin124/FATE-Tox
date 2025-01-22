import os
import pandas as pd
import numpy as np
import torch
import sklearn
import logging
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem, rdchem, BRICS, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MolFromSmiles, FragmentOnBonds, RDConfig, FragmentCatalog
from rdkit.Avalon import pyAvalonTools

from sklearn.metrics import pairwise_distances
from collections import defaultdict
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils.smiles import from_smiles
from copy import deepcopy
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_scatter import scatter


from data_utils_modified_2d import *
from dataset_construction_2d import * 
from utils import *
from split import *

import argparse

##### bring dataset to dataloader #####
def bring_dataset_info(args):
    if args.task_name in ['Skin_Reaction','Carcinogens_Lagunin', 'herg']:
        with open(f'/data/project/sumin/toxicity/mat_egnn/data/tdc/{args.task_name}_pos_idx.pkl', 'rb') as f:
            data = pickle.load(f)
        args.smiles_col = 'Drug'
        args.target_col = 'Y'
        args.num_task = 1
    
    elif args.task_name in ['sider', 'clintox', 'bbbp']:
        data = pd.read_pickle(f'/data/project/sumin/toxicity/mat_egnn/data/{args.task_name}_pos.pkl')
        if args.task_name in ['sider', 'clintox']:
            args.target_col, args.smiles_col = [col for col in data.columns if col not in ['smiles', 'mol']], 'smiles'
            args.num_task = len(args.target_col)
        elif args.task_name == 'bbbp':
            args.target_col, args.smiles_col = 'p_np', 'smiles'
            args.num_task = 1
    if args.num_task != 1: 
        args.is_multitask = True
    
    return args, data

def return_name(args):
    group_name = f'2d_dist_{args.num_layers}_do{args.clf_dropout}_{args.task_name}_{args.task}_h{args.hdim}l{args.GNN_layers}k{args.knn}c{args.cdim}_Mh{args.MAT_heads}l{args.MAT_layers}_do_{args.dropout}_lambda{args.lambda_attention}_{args.lambda_distance}_b{args.batch_size}lr{args.lr}w{args.weight_decay}'
    model_name = f'{args.fragment}_{args.ablation}_updatex_{group_name}'
    return group_name, model_name
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=int, default=0)
#    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--n_epochs', type = int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--ablation', type=str, default=None)
    parser.add_argument('--coor_normalize', type=bool, default=True)
    parser.add_argument('--ecfp_dim', type=int, default=1024)
    parser.add_argument('--ecfp_radius', type=int, default=2)
    ###fragmentation
    parser.add_argument('--brics_weight', type=float, default=0.33)
    parser.add_argument('--fg_weight', type=float, default=0.33)
    parser.add_argument('--frozen_weight', type=bool, default=False)    
    
    parser.add_argument('--hdim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--clf_dropout', type=float, default=0.0)
    
    # egnn params
    parser.add_argument('--cdim', type=int, default=32)
    parser.add_argument('--knn', type=int, default=48)
    parser.add_argument('--GNN_layers', type=int, default=3)
    parser.add_argument('--num_r_gaussian', type=int, default=64)
    
    # 2D Transformer params
    parser.add_argument('--MAT_heads', type=int, default=8)
    parser.add_argument('--MAT_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lambda_attention', type=float, default=0.6)
    parser.add_argument('--lambda_distance', type=float, default=0.2)
    parser.add_argument('--mat_agg', type=str, default='mean')
    parser.add_argument('--init_type', type=str, default = 'uniform')
    parser.add_argument('--task_name', type=str, default='DILI')
    parser.add_argument('--dataset', type=str, default='moleculenet')
    
    args = parser.parse_args()
    
    args.MAT_hdim = args.hdim 
    args.is_multitask = False
    args.d_atom = 33


    from models.fate_tox import *
    from train_ import *
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

    args, data = bring_dataset_info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    
    train_dataset = MultiFragDataset_w_fp(data.iloc[idx_list['train']], args.smiles_col, args.target_col, 'train', args.coor_normalize, args)
    val_dataset = MultiFragDataset_w_fp(data.iloc[idx_list['val']], args.smiles_col, args.target_col, 'val', args.coor_normalize, args)
    test_dataset = MultiFragDataset_w_fp(data.iloc[idx_list['test']], args.smiles_col, args.target_col, 'test', args.coor_normalize, args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

### bring model
    model = fate_tox_model(args, output_dim=1, tasks=args.num_task).to(args.device)
    optimizer= torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss() 

    early_stopper=EarlyStopper(patience=args.patience,printfunc=logger,verbose=True,path=f'ckpts/{logger.date}_{logger.model_name}.pt')

### start training
    epoch = 0
    
    test_auroc_list = []
    while epoch<args.n_epochs:
        epoch += 1
        train_loss = train_multitask(model, optimizer, train_loader, args, criterion)
        valid_loss, val_aucroc_tasks = eval_multitask(model, val_loader, args, criterion)
        test_loss, test_auroc_tasks = eval_multitask(model, test_loader, args, criterion)
        val_auroc = np.mean(val_aucroc_tasks)
        test_auroc = np.mean(test_auroc_tasks)
        test_auroc_list.append(test_auroc)
    
        logger(f'[Epoch{epoch}] train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, valid auroc: {val_auroc:.4f}')
        logger(f'test_loss: {test_loss:.4f}, test_auroc: {test_auroc:.4f}')
        
        early_stopper(-val_auroc,epoch, model)
        if early_stopper.early_stop:
            logger('early stopping')
            break 

    if args.task == 'classification':
        logger(f'early stopped: loaded best model "{early_stopper.path}",\n \
                valid loss: {early_stopper.val_loss_min:.4f}, valid auroc: {early_stopper.best_score:.4f}\
                test auroc: {test_auroc_list[early_stopper.best_epoch-1]:.4f}')

    elif args.task == 'regression':
        logger(f'early stopped: loaded best model "{early_stopper.path}",\n \
                valid loss: {early_stopper.val_loss_min:.4f}, valid rmse: {early_stopper.best_score:.4f}\
                test rmse: {test_auroc_list[early_stopper.best_epoch-1]:.4f}')




