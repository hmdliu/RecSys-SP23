
import os
import pickle
import random
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
from collections import defaultdict

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

num_ng = 3
total_epochs = 20

def set_seed(seed=42):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_all_custom(test_num=100):
    train_data = pd.read_csv('./train_df')    
    
    train_data = train_data[['uid', 'sid']]
    train_data['uid'] = train_data['uid'].apply(lambda x : int(x))
    train_data['sid'] = train_data['sid'].apply(lambda x : int(x))    
    train_data.columns = ['user', 'item']
    
    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    total_mat = train_mat

    return train_data, user_num, item_num, train_mat, total_mat

class BPRData(data.Dataset):
    def __init__(self, features, 
                num_item, train_mat=None, total_mat=None, num_ng=1, is_training=None):
        super(BPRData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.total_mat = total_mat
        self.num_ng = num_ng
        self.is_training = is_training

        self.feat_num = len(features)
        self.full_set = set(range(200))
        self.pos_dict = defaultdict(set)
        self.neg_dict = defaultdict(set)
        for u, i in map(tuple, features):
            self.pos_dict[u].add(i)
        for u in self.pos_dict:
            self.neg_dict[u] = list(self.full_set - self.pos_dict[u])
        for k, v in self.pos_dict.items():
            self.pos_dict[k] = list(v)

    def __len__(self):
        return self.num_ng * len(self.features) if self.is_training else len(self.features)

    def __getitem__(self, idx):
        u = random.randint(0, 199)
        pos1 = random.choice(self.pos_dict[u])
        pos2 = random.choice(self.pos_dict[u])
        neg1 = random.choice(self.neg_dict[u])
        neg2 = random.choice(self.neg_dict[u])
        return [u, pos1, pos2, neg1, neg2]
    
    def fetch_data(self):
        return [self[i] for i in range(len(self))]

if __name__ == '__main__':

    set_seed(0)

    train_data, user_num, item_num, train_mat, total_mat = load_all_custom()
    print('original user-pos tuple is')
    print(train_data[0:5])

    train_dataset = BPRData(train_data, item_num, train_mat, total_mat, num_ng=num_ng, is_training=True)
    print('new (user, pos1, pos2, neg1, neg2) tuple is')
    print(train_dataset[0:5])

    os.makedirs('./train_samples', exist_ok=True)
    for i in range(total_epochs):
        print('epoch', i)
        train_list = train_dataset.fetch_data()
        with open(f'./train_samples/train_samples_{i}', 'wb') as fp:
            pickle.dump(train_list, fp)
