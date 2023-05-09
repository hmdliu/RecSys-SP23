
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

def heatmap(model_path, save_path):

    model = torch.load(model_path)
    model.eval()
    model.cuda()
    user_emb = model.embed_user_MLP.weight.detach().cpu()
    item_emb = model.embed_item_MLP.weight.detach().cpu()
    user_num = user_emb.shape[0]
    item_num = item_emb.shape[0]
    pred_mtx = np.zeros((user_num, item_num))

    for user in range(user_num):
        pos_score, _ = model(torch.tensor([user]*item_num).cuda(), torch.tensor(list(range(item_num))).cuda(), torch.tensor(list(range(item_num))).cuda())
        pred_mtx[user,:] = pos_score.cpu().detach()
        
    ax = plt.axes()
    sns.heatmap(pred_mtx)
    ax.set_xlabel('Item Index')
    ax.set_ylabel('User Index')
    ax.xaxis.set_ticks([0, 25, 50, 75, 100, 125, 150, 175, 200], [0, 25, 50, 75, 100, 125, 150, 175, 200])
    ax.yaxis.set_ticks([0, 25, 50, 75, 100, 125, 150, 175, 200], [0, 25, 50, 75, 100, 125, 150, 175, 200])

    plt.savefig(save_path)

heatmap('./models/final_synthetic1__MF_none_0.8_10.pth', './fig/none.png')
heatmap('./models/final_synthetic1__MF_posneg_0.8_10.pth', './fig/posneg.png')
heatmap('./models/final_synthetic1__MF_pos2neg2_0.8_10.pth', './fig/pos2neg2.png')
    