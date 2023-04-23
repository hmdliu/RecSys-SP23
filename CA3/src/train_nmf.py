
import os
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .utils import *
from .model import NMF
from .dataset import NMFDataset

def train_nmf(configs, verbose=True):

    # set random seed
    set_seed(seed=42)

    # init variables
    train_loss = 0
    valid_loss = 0
    train_losses = []
    valid_losses = []
    best_loss = np.inf

    # init model
    device = configs['device']
    model = NMF(configs['model_config']).to(device)

    # init loss function
    loss_fn = nn.BCELoss(reduction='sum')

    # init dataloaders
    with open(configs['split_path'], 'rb') as f:
        tp = pickle.load(f)
    train_ds, valid_ds, _ = tuple([NMFDataset(df) for df in tp])
    train_dl = DataLoader(train_ds, batch_size=configs['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=configs['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    # init optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, len(train_dl) * configs['epochs'], eta_min=1e-6)

    # train for a given amount of epochs
    for e in range(configs['epochs']): 
        
        # start batch training
        model.train()
        for u, i, y in train_dl:
            # send data to device
            u, i, y = u.to(device), i.to(device), y.to(device)
            # model prediction
            r_pred = model(u, i)
            # compute loss
            loss = loss_fn(r_pred, y)
            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.detach().item()
    
        # record loss
        train_loss /= len(train_ds)     # average loss over all examples.
        train_losses.append(train_loss)
        
        # validation
        model.eval()                    # this fix the trainable parameters.
        for u, i, y in valid_dl:
            u, i, y = u.to(device), i.to(device), y.to(device)
            r_pred = model(u, i)
            loss = loss_fn(r_pred, y)
            valid_loss += loss.detach().item()
    
        valid_loss /= len(valid_ds)
        valid_losses.append(valid_loss)
        
        if verbose and (e % 10 == 0):
            print(f'Epoch {e} Train loss: {train_loss:.4f}; Valid loss: {valid_loss:.4f}')

        if valid_loss < best_loss:
            best_loss = valid_loss
            ckpt_dict = {
                'epoch': e,
                'valid_loss': valid_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            ckpt_path = os.path.join(configs['save_path'], f"{configs['search_id']}.ckpt")
            torch.save(ckpt_dict, ckpt_path)
    
    return train_losses, valid_losses

if __name__ == '__main__':
    rmse_list = []
    search_num = 10
    for s in range(search_num):
        configs = {
            'search_id': s,
            # dataset config
            'split_path': './pkl/split_loo.pkl',
            # training config
            'lr': log_uniform(1e-4, 1e-2, 4),
            'epochs': 100,
            'batch_size': 2048,
            'weight_decay': log_uniform(1e-4, 1e-2, 4),
            'save_path': f'./results/hps_nmf_loo',
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            # model config
            'model_config': {
                'user_cnts': 6041,      # unique users for embedding layer
                'item_cnts': 3953,      # unique items for embedding layer
                'emb_dim': 10,          # size of embedding vector
                'hidden_dims': (10, ),  # hidden dims tuple
            }
        }
        print(f'\n==> Hyper-parameter Search {s}\nConfigs: {configs}')
        os.makedirs(configs['save_path'], exist_ok=True)
        train_losses, valid_losses = train_nmf(configs, verbose=True)
        rmse_list.append((s, round(np.sqrt(min(valid_losses)), 4)))
        print(f'Valid RMSE = {rmse_list[-1][1]}')
    rmse_list.sort(key=lambda x: x[1])
    print(f'\nValid RMSE list: {rmse_list}')
    
