
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

def load_dfs():
    rating_df = pd.read_csv("ml-1m/ratings.dat", sep='::', header=None, engine="python", names=["userID", "movieID", "rating", "timestamp"])
    movie_df = pd.read_csv("ml-1m/movies.dat", sep='::', header=None, engine="python", encoding='latin-1', names=["movieID", "title", "genre"])
    user_df = pd.read_csv("ml-1m/users.dat", sep='::', header=None, engine="python", names=["userID", "gender", "age", "occupation", "zipcode"])
    return rating_df, movie_df, user_df

def loo_split():
    try:
        with open('pkl/split_loo.pkl', 'rb') as f:
            tp = pickle.load(f)
        print('Pre-computed LOO split loaded')
    except:
        train_list, valid_list, test_list = [], [], []
        rating_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine="python", names=["userID", "movieID", "rating", "timestamp"])
        # rating_df['rating'] = rating_df['rating'].apply(lambda x: 1 if x > 3 else 0)
        user_list = rating_df['userID'].unique()
        for u in tqdm(user_list):
            df = rating_df[rating_df['userID'] == u].sort_values(by='timestamp')
            train_list.append(df.iloc[:-2])
            valid_list.append(df.iloc[-2:-1])
            test_list.append(df.iloc[-1:])
        tp = pd.concat(train_list), pd.concat(valid_list), pd.concat(test_list), rating_df
        os.makedirs('./pkl', exist_ok=True)
        with open('./pkl/split_loo.pkl', 'wb') as f:
            pickle.dump(tp, f)
        print('LOO split has been saved to ./pkl/split_loo.pkl')
    return tp

def hp_split(ratio=[8, 1, 1]):
    try:
        with open('pkl/split_hp.pkl', 'rb') as f:
            tp = pickle.load(f)
        print('Pre-computed HP split loaded')
    except:
        assert len(ratio) == 3
        train_list, valid_list, test_list = [], [], []
        rating_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine="python", names=["userID", "movieID", "rating", "timestamp"])
        # rating_df['rating'] = rating_df['rating'].apply(lambda x: 1 if x > 3 else 0)
        user_list = rating_df['userID'].unique()
        for u in tqdm(user_list):
            df = rating_df[rating_df['userID'] == u].sort_values(by='timestamp')
            r1 = int(len(df) * ratio[0] / sum(ratio))
            r2 = int(len(df) * (ratio[0] + ratio[1]) / sum(ratio))
            train_list.append(df.iloc[:r1])
            valid_list.append(df.iloc[r1:r2])
            test_list.append(df.iloc[r2:])
        tp = pd.concat(train_list), pd.concat(valid_list), pd.concat(test_list), rating_df
        os.makedirs('./pkl', exist_ok=True)
        with open('./pkl/split_hp.pkl', 'wb') as f:
            pickle.dump(tp, f)
        print('HP split has been saved to ./pkl/split_hp.pkl')
    return tp

class PosNegDataset(Dataset):
    def __init__(self, dfs, idx=0, user_col=0, item_col=1, neg_samples=1, eval_flag=False):
        # init variables
        df, df_all = dfs[idx], dfs[-1]
        self.df = dfs[idx].reset_index()
        self.eval_flag = eval_flag
        self.neg_samples = neg_samples
        self.user_tensor = torch.tensor(self.df.iloc[:, user_col+1], dtype=torch.long)
        self.item_tensor = torch.tensor(self.df.iloc[:, item_col+1], dtype=torch.long)
        # pre-compute pos/neg item set
        self.user_list = np.unique(df_all.iloc[:, user_col].values)
        self.item_list = np.unique(df_all.iloc[:, item_col].values)
        self.item_set_all = set(np.unique(df_all['movieID'].values))
        self.pis_dict, self.nis_dict = {}, {}
        for u in self.user_list:
            pis_all = set(df_all[df_all['userID'] == u.item()]['movieID'].values)
            self.nis_dict[u] = self.item_set_all - pis_all      # neg item set for sampling
            if self.eval_flag:
                pis_eval = set(df[df['userID'] == u.item()]['movieID'].values)
                self.pis_dict[u] = pis_eval                     # gt item set for evaluation      

    def __getitem__(self, index):
        if self.eval_flag:
            y = self.pis_dict[self.user_list[index]]
            i_neg = self.nis_dict[self.user_list[index]]
            i_all = torch.tensor(list(i_neg) + list(y), dtype=torch.long)
            i_sub = torch.tensor(random.sample(i_neg, k=100) + list(y), dtype=torch.long)
            u_all = torch.full_like(i_all, fill_value=self.user_list[index])
            u_sub = torch.full_like(i_sub, fill_value=self.user_list[index])
            return u_all, i_all, u_sub, i_sub, y
        else:
            u, i = self.user_tensor[index], self.item_tensor[index]
            js = random.sample(self.nis_dict[u.item()], k=self.neg_samples)
            js = torch.tensor(js, dtype=torch.long)
            return u, i, js

    def __len__(self):
        return len(self.user_list) if self.eval_flag else self.item_tensor.shape[0]

if __name__ == '__main__':

    print('\nLoading dataframes...')
    outputs = [str(df.head()) for df in load_dfs()]
    print('\n'.join(outputs))

    print('\nLoading data using LOO and HP strategy...')
    loo_split()
    hp_split()
