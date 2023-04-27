
import os
import sys
import torch
import pickle

from .utils import *
from .model import NMF
from .dataset import PosNegDataset
from .metrics import compute_metrics

def test_nmf(configs):

    # init model
    device = configs['device']
    model = NMF(configs['model_config']).to(device)
    print(f'\nModel: {model}')

    # load checkpoint
    assert os.path.exists(configs['resume'])
    ckpt = torch.load(configs['resume'])
    model.load_state_dict(ckpt['state_dict'])

    # init dataloaders
    with open(configs['split_path'], 'rb') as f:
        tp = pickle.load(f)
    test_ds = PosNegDataset(tp, idx=2, neg_samples=0, eval_flag=True)
        
    # validation
    model.eval()
    all_list, sub_list = [], []
    for i in range(len(test_ds)):
        # fetch one user & send data to device
        u_all, i_all, u_sub, i_sub, y = test_ds[i]
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

    # logging
    print(f'Test metrics (all): {metrics_all}')
    print(f'Test metrics (sub): {metrics_sub}')

if __name__ == '__main__':
    assert sys.argv[1] in ('hp', 'loo')
    if not os.path.exists('./pkl'):
        init_data_split()
    configs = {
        # dataset config
        'split_path': f'./pkl/split_{sys.argv[1]}.pkl',
        # test config
        'resume': f'./results/hps_nmf_{sys.argv[1]}/best.ckpt',
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        # model config
        'model_config': {
            'user_cnts': 6041,      # unique users for embedding layer
            'item_cnts': 3953,      # unique items for embedding layer
            'emb_dim': 10,          # size of embedding vector
            'hidden_dims': (10, ),  # hidden dims tuple
        }
    }
    print(f'\nConfigs: {configs}\n\n==> Testing start')
    test_nmf(configs)
