import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data
import os
os.chdir('./')

import random as random

random.seed(0)

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
                num_item, train_mat=None, total_mat=None, num_ng=0, is_training=None, sample_mode = None):
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

    def ng_sample(self):
        if True:
            assert self.is_training, 'no need to sampling when testing'
            self.features_fill = []

            tmp = pd.DataFrame(self.features)
            tmp.columns = ['uid', 'sid']
            
            tmp = tmp.sort_values('uid')
            tmp_list = list(range(tmp.shape[0]))
            random.shuffle(tmp_list)
            tmp['rng'] = tmp_list
            sid2 = tmp.sort_values(['uid', 'rng']).sid
            tmp['sid2'] = sid2.reset_index().sid
            tmp = tmp[['uid', 'sid', 'sid2']]
            tmp = tmp.sort_index()
            self.features2 = tmp.values.tolist()         
                
        for x in self.features2:
            u, pos1, pos2 = x[0], x[1], x[2]
            for t in range(self.num_ng):
                if u == 0:
                    neg1, neg2 = 199, 199                    
                elif u == 1:
                    neg1, neg2 = 198, 198                    
                elif u == 199:
                    neg1, neg2 = np.random.randint(199-u + 1, 200, size = 2)
                else:
                    neg1, neg2 = np.random.randint(199-u, 200, size = 2)
                self.features_fill.append([u, pos1, pos2, neg1, neg2])
    def __len__(self):
        return self.num_ng * len(self.features) if self.is_training \
                    else len(self.features)
    def __getitem__(self, idx):
        features = self.features_fill if \
                    self.is_training else self.features
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if \
                    self.is_training else features[idx][1]        
        return user, item_i, item_j

train_data, user_num, item_num, train_mat, total_mat = load_all_custom()
print('original user-pos tuple is')
print(train_data[0:10])

train_dataset = BPRData(train_data, item_num, train_mat, total_mat, num_ng=1, is_training=True, sample_mode=None)

train_dataset.ng_sample()
negative_samples = train_dataset.features_fill
print('new (user, pos1, pos2, neg1, neg2) tuple is')
print(negative_samples[0:10])

tmp1 = np.array(negative_samples)[:, 1]
tmp2 = np.array(negative_samples)[:, 2]
print('ratio of pos1 > pos2')
print(np.mean(tmp1 > tmp2))

random.seed(0)
total_epochs = 20
num_ng = 3

import pickle
for i in range(total_epochs):
    print(i)
    train_list = []
    for j in range(num_ng):
        train_dataset.ng_sample()
        train_samples = train_dataset.features_fill
        train_list += train_samples
    with open(f'./train_samples/train_samples_{i}', 'wb') as fp:
        pickle.dump(train_list, fp)
