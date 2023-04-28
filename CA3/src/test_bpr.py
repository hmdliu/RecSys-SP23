
import os
import sys
import time
import torch
import pickle

from .utils import *
from .model import BPR
from .dataset import PosNegDataset
from .metrics import compute_metrics

def test_bpr(configs):

    # init model
    device = configs['device']
    model = BPR(configs['model_config']).to(device)
    print(f'\nModel: {model}')

    # load checkpoint
    ckpt = torch.load(configs['resume_path'])
    model.load_state_dict(ckpt['state_dict'])

    # init dataloaders
    with open(configs['split_path'], 'rb') as f:
        tp = pickle.load(f)
    test_ds = PosNegDataset(tp, idx=2, neg_samples=0, eval_flag=True)
        
    # init testing
    model.eval()
    all_list, sub_list = [], []

    # full corpus
    start_time = time.time()
    for i in range(len(test_ds)):
        # fetch one user & send data to device
        u_all, i_all, u_sub, i_sub, y = test_ds[i]
        u_all, i_all = u_all.to(device), i_all.to(device)
        # model prediction
        with torch.no_grad():
            r_pred_all = model.predict(u_all, i_all)
        # convert probas to ordered item ids
        i_pred_all = i_all[torch.argsort(r_pred_all, descending=True)]
        all_list.append((i_pred_all, y))
    metrics_all = compute_metrics(all_list, round_digits=4)
    total_time = time.time() - start_time
    print(f'Test metrics (all): {metrics_all} | Time {total_time:.2f}s')

    # sampled corpus
    start_time = time.time()
    for i in range(len(test_ds)):
        # fetch one user & send data to device
        u_all, i_all, u_sub, i_sub, y = test_ds[i]
        u_sub, i_sub = u_sub.to(device), i_sub.to(device)
        # model prediction
        with torch.no_grad():
            r_pred_sub = model.predict(u_sub, i_sub)
        # convert probas to ordered item ids
        i_pred_sub = i_sub[torch.argsort(r_pred_sub, descending=True)]
        sub_list.append((i_pred_sub, y))
    metrics_sub = compute_metrics(sub_list, round_digits=4)
    total_time = time.time() - start_time
    print(f'Test metrics (sub): {metrics_sub} | Time {total_time:.2f}s')

if __name__ == '__main__':
    split_path = f'./pkl/split_{sys.argv[1]}.pkl'
    resume_path = f'./results/hps_bpr_{sys.argv[1]}/best.ckpt'
    if not os.path.exists('./pkl'):
        init_data_split()
    assert os.path.exists(split_path) and os.path.exists(resume_path)
    configs = {
        # dataset config
        'split_path': split_path,
        # test config
        'resume_path': resume_path,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # model config
        'model_config': {
            'user_cnts': 6041,      # unique users for embedding layer
            'item_cnts': 3953,      # unique items for embedding layer
            'emb_dim': 10,          # size of embedding vector
        }
    }
    print(f'\nConfigs: {configs}\n\n==> Testing start')
    test_bpr(configs)
