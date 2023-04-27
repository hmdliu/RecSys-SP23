
import os
import sys
import pickle
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .utils import *
from .model import NMF
from .dataset import PosNegDataset
from .metrics import compute_metrics

def train_nmf(configs, verbose=True):

    # init variables
    train_losses = []
    metrics_list_all = []
    metrics_list_sub = []
    train_loss = es_patience = 0
    best_pred_all = best_pred_sub = -1

    # init model
    device = configs['device']
    model = NMF(configs['model_config']).to(device)
    print(f'\nModel: {model}')

    # init loss function
    loss_fn = nn.BCELoss(reduction='sum')

    # init dataloaders
    with open(configs['split_path'], 'rb') as f:
        tp = pickle.load(f)
    train_ds, valid_ds = PosNegDataset(tp, idx=0, eval_flag=False), PosNegDataset(tp, idx=1, eval_flag=True)
    train_dl = DataLoader(train_ds, batch_size=configs['batch_size'], shuffle=True, num_workers=2, pin_memory=True)

    # init optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, len(train_dl) * configs['epochs'], eta_min=1e-6)

    # train for a given amount of epochs
    for e in range(configs['epochs']): 
        
        # start batch training
        count = 0
        model.train()
        for u, p, n in train_dl:
            # concat pos & neg samples
            n = n.flatten()
            i = torch.cat((p, n), dim=0)
            u = u.repeat(i.shape[0] // u.shape[0])
            y = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])), dim=0)
            count += i.shape[0]
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
    
        # record loss (average loss over all examples)
        train_loss /= count
        train_losses.append(train_loss)
        
        # validation
        model.eval()
        all_list, sub_list = [], []
        for i in range(len(valid_ds)):
            # fetch one user & send data to device
            u_all, i_all, u_sub, i_sub, y = valid_ds[i]
            u_all, i_all = u_all.to(device), i_all.to(device)
            u_sub, i_sub = u_sub.to(device), i_sub.to(device)
            # model prediction
            with torch.no_grad():
                r_pred_all = model(u_all, i_all)
                r_pred_sub = model(u_sub, i_sub)
            # convert probas to ordered item ids
            i_pred_all = i_all[torch.argsort(r_pred_all, descending=True)]
            i_pred_sub = i_sub[torch.argsort(r_pred_sub, descending=True)]
            all_list.append((i_pred_all, y))
            sub_list.append((i_pred_sub, y))
        
        # compute metrics
        metrics_all = compute_metrics(all_list, round_digits=4)
        metrics_sub = compute_metrics(sub_list, round_digits=4)
        metrics_list_all.append(metrics_all)
        metrics_list_sub.append(metrics_sub)

        # update best pred & save checkpoints
        pred_all, pred_sub = sum(metrics_all.values()), sum(metrics_sub.values())
        if pred_all > best_pred_all:
            best_pred_all = pred_all
            ckpt_dict = {
                'epoch': e,
                'metrics': metrics_all,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            ckpt_path = os.path.join(configs['save_path'], f"all_{configs['search_id']}.ckpt")
            torch.save(ckpt_dict, ckpt_path)
        if pred_sub > best_pred_sub:
            best_pred_sub = pred_sub
            ckpt_dict = {
                'epoch': e,
                'metrics': metrics_sub,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            ckpt_path = os.path.join(configs['save_path'], f"sub_{configs['search_id']}.ckpt")
            torch.save(ckpt_dict, ckpt_path)
        
        # logging
        if verbose or (e % 5 == 0):
            print(f'Epoch {e} | Train loss: {train_loss:.4f}')
            print(f'Epoch {e} | Valid metrics (all): {metrics_all}')
            print(f'Epoch {e} | Valid metrics (sub): {metrics_sub}')

        # early stopping
        if len(metrics_list_all) >= 2:
            curr = sum_metrics(metrics_list_all[-1])
            prev = sum_metrics(metrics_list_all[-2])
            es_patience = 0 if curr >= prev else (es_patience + 1)
            if es_patience >= 5:
                print('Early stopping triggered')
                break
    
    return train_losses, metrics_list_all, metrics_list_sub

if __name__ == '__main__':
    vl_list = []
    search_num = 5
    split_path = f'./pkl/split_{sys.argv[1]}.pkl'
    if not os.path.exists('./pkl'):
        init_data_split()
    assert os.path.exists(split_path)
    configs = {
        # dataset config
        'split_path': f'./pkl/split_{sys.argv[1]}.pkl',
        # training config
        'lr': 'TBD',
        'epochs': 100,
        'batch_size': 2048,
        'weight_decay': 'TBD',
        'save_path': f'./results/hps_nmf_{sys.argv[1]}',
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # model config
        'model_config': {
            'user_cnts': 6041,      # unique users for embedding layer
            'item_cnts': 3953,      # unique items for embedding layer
            'emb_dim': 10,          # size of embedding vector
            'hidden_dims': (10, ),  # hidden dims tuple
        }
    }
    print(f'\nBase configs: {configs}')
    if os.path.exists(configs['save_path']):
        print(f'\nRemoving exsiting save_path ...')
        shutil.rmtree(configs['save_path'])
    os.makedirs(configs['save_path'])
    for s in range(search_num):
        to_modify = {
            'search_id': s,
            'lr': log_uniform(1e-3, 1e-1, 3),
            'weight_decay': log_uniform(1e-3, 1e-1, 3)
        }
        configs.update(to_modify)
        print(f'\n==> Hyper-parameter Search {s}\nModified configs: {to_modify}')
        train_losses, metrics_list_all, metrics_list_sub = train_nmf(configs, verbose=True)
        vl_list.append((s, max(metrics_list_all, key=sum_metrics)))
        print(f'Valid metrics (all): {vl_list[-1][1]}')
    vl_list.sort(key=lambda x: sum_metrics(x[1]), reverse=True)
    shutil.copyfile(
        src=os.path.join(configs['save_path'], f'all_{vl_list[0][0]}.ckpt'),
        dst=os.path.join(configs['save_path'], 'best.ckpt')
    )
    print(f'\nBest pred: {vl_list[0]}')
