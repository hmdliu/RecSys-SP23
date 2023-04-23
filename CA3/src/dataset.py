
import pickle
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
        rating_df['rating'] = rating_df['rating'].apply(lambda x: 1 if x > 3 else 0)
        user_list = rating_df['userID'].unique()
        for u in tqdm(user_list):
            df = rating_df[rating_df['userID'] == u]
            train_list.append(df.iloc[:-2])
            valid_list.append(df.iloc[-2:-1])
            test_list.append(df.iloc[-1:])
        tp = pd.concat(train_list), pd.concat(valid_list), pd.concat(test_list)
        with open('pkl/split_loo.pkl', 'wb') as f:
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
        rating_df['rating'] = rating_df['rating'].apply(lambda x: 1 if x > 3 else 0)
        user_list = rating_df['userID'].unique()
        for u in tqdm(user_list):
            df = rating_df[rating_df['userID'] == u]
            r1 = int(len(df) * ratio[0] / sum(ratio))
            r2 = int(len(df) * (ratio[0] + ratio[1]) / sum(ratio))
            train_list.append(df.iloc[:r1])
            valid_list.append(df.iloc[r1:r2])
            test_list.append(df.iloc[r2:])
        tp = pd.concat(train_list), pd.concat(valid_list), pd.concat(test_list)
        with open('pkl/split_hp.pkl', 'wb') as f:
            pickle.dump(tp, f)
        print('HP split has been saved to ./pkl/split_hp.pkl')
    return tp

class NMFDataset(Dataset):
    def __init__(self, df, user_col=1, item_col=2, rating_col=3):
        self.df = df.reset_index()
        self.user_tensor = torch.tensor(self.df.iloc[:, user_col], dtype=torch.long)
        self.item_tensor = torch.tensor(self.df.iloc[:, item_col], dtype=torch.long)
        self.target_tensor = torch.tensor(self.df.iloc[:, rating_col], dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.target_tensor.shape[0]

if __name__ == '__main__':

    print('\nLoading dataframes...')
    outputs = [str(df.head()) for df in load_dfs()]
    print('\n'.join(outputs))

    print('\nLoading data using LOO and HP strategy...')
    loo_split()
    hp_split()
