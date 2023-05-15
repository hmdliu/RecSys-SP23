import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

raw = pd.read_csv('./rawdata/ratings.dat', sep = "::", header = None)
raw.columns = ['uid', 'sid', 'ratings', 'timestamp']
raw = raw[['uid', 'sid']]
new_items = raw.sid.value_counts()[raw.sid.value_counts() >= 10].reset_index()['index'].values
data2 = raw[raw.sid.isin(new_items)]
data2 = data2.reset_index()[['uid', 'sid']]

# reindex by popularity count
pop_uid = data2.uid.value_counts().reset_index()
pop_uid.columns = ['uid', 'uid_counts']
pop_uid_dict = pop_uid.reset_index()
pop_uid_dict = pop_uid_dict[['index', 'uid']]
pop_uid_dict.columns = ['new_uid', 'uid']
pop_uid_dict = dict(zip(pop_uid_dict.uid, pop_uid_dict.new_uid))
pop_sid = data2.sid.value_counts().reset_index()
pop_sid.columns = ['sid', 'sid_counts']
pop_sid_dict = pop_sid.reset_index()
pop_sid_dict = pop_sid_dict[['index', 'sid']]
pop_sid_dict.columns = ['new_sid', 'sid']
pop_sid_dict = dict(zip(pop_sid_dict.sid, pop_sid_dict.new_sid))
data2['uid'] = data2.uid.map(pop_uid_dict).values
data2['sid'] = data2.sid.map(pop_sid_dict).values
data2 = data2.sort_values(['uid', 'sid'], ascending = [True, True])
tmp = data2
tmp['one'] = 1
data2 = data2.reset_index()[['uid', 'sid']]
total_data = data2
total_data = total_data.reset_index()[['uid', 'sid']]

# test data split
random.seed(0)
train_df = total_data.sample(frac = 0.8, random_state = 0)
test_df = total_data.loc[list(set(total_data.index) - set(train_df.index))]
print('=' * 64)
print(len(total_data.uid.unique()))
print(len(total_data.sid.unique()))
print('=' * 64)
print(len(train_df.uid.unique()))
print(len(train_df.sid.unique()))
print('=' * 64)
print(len(test_df.uid.unique()))
print(len(test_df.sid.unique()))
print('=' * 64)
print(total_data.uid.value_counts().tail())
print(total_data.sid.value_counts().tail())
print('=' * 64)
print(train_df.uid.value_counts().tail())
print(train_df.sid.value_counts().tail())
print('=' * 64)
print(test_df.uid.value_counts().tail())
print(test_df.sid.value_counts().tail())
print('=' * 64)
print(train_df.head())
print('=' * 64)
print(test_df.head())

# make test negative sample
random.seed(0)
n_user = len(total_data.uid.unique())
n_item = len(total_data.sid.unique())
item_set = set(list(range(n_item)))
neg_sample_df = pd.DataFrame({'uid' : [], 'sid' : []})
for user in list(range(n_user)):
    true_set = total_data[total_data['uid'] == user]['sid'].values
    true_set = set(true_set)
    user_neg_samples = item_set - true_set
    user_neg_samples = list(user_neg_samples)
    list_len = len(user_neg_samples)
    user_neg_samples = random.sample(user_neg_samples, 100)
    tmp_neg_sample_df = pd.DataFrame({'uid' : [user]*100, 'sid' : user_neg_samples})
    neg_sample_df = pd.concat([neg_sample_df, tmp_neg_sample_df])
print('=' * 64)
print(neg_sample_df.head())

neg_sample_df['uid'] = neg_sample_df['uid'].astype(int)
neg_sample_df['sid'] = neg_sample_df['sid'].astype(int)
test_df['type'] = 'pos'
neg_sample_df['type'] = 'neg'
print('=' * 64)
print(test_df.head())

test_neg_sample_df = neg_sample_df.copy()
test_df_with_neg = pd.concat([test_df, neg_sample_df])
print('=' * 64)
print(test_df_with_neg.head())

train_df = train_df.reset_index()[['uid', 'sid']]
test_df = test_df.reset_index()[['uid', 'sid', 'type']]
neg_sample_df = neg_sample_df.reset_index()[['uid', 'sid', 'type']]
test_df_with_neg = test_df_with_neg.reset_index()[['uid', 'sid', 'type']]
print('=' * 64)
print(train_df.head())

# val data split
real_train_df = train_df.sample(frac = 0.75, random_state = 0)
val_df = train_df.loc[list(set(train_df.index) - set(real_train_df.index)) ]
print('=' * 64)
print(real_train_df.head())
print('=' * 64)
print(val_df.head())

val_df['type'] = 'pos'
val_df_with_neg = pd.concat([val_df, neg_sample_df])
print('=' * 64)
print(val_df_with_neg.head())

real_train_df = real_train_df.reset_index()[['uid', 'sid']]
val_df = val_df.reset_index()[['uid', 'sid', 'type']]
val_df_with_neg = val_df_with_neg.reset_index()[['uid', 'sid', 'type']]

# summarize
test_df = test_df[['uid', 'sid', 'type']]
test_df_with_neg = test_df_with_neg[['uid', 'sid', 'type']]
assert total_data.shape[0] == real_train_df.shape[0] + val_df.shape[0] + test_df.shape[0]
assert val_df_with_neg.shape[0] == val_df.shape[0] + neg_sample_df.shape[0]
assert test_df_with_neg.shape[0] == test_df.shape[0] + neg_sample_df.shape[0]

total_data.to_csv('total_df', index = False)
real_train_df.to_csv('train_df', index = False)
neg_sample_df.to_csv('neg_sample_df', index = False)
val_df.to_csv('val_df', index = False)
val_df_with_neg.to_csv('val_df_with_neg', index = False)
test_df.to_csv('test_df', index = False)
test_df_with_neg.to_csv('test_df_with_neg', index = False)

uid_pop_total = total_data.uid.value_counts().reset_index()
uid_pop_total.columns = ['uid', 'total_counts']
sid_pop_total = total_data.sid.value_counts().reset_index()
sid_pop_total.columns = ['sid', 'total_counts']

uid_pop_train = train_df.uid.value_counts().reset_index()
uid_pop_train.columns = ['uid', 'train_counts']
sid_pop_train = train_df.sid.value_counts().reset_index()
sid_pop_train.columns = ['sid', 'train_counts']

uid_pop_total.to_csv('uid_pop_total', index = False)
sid_pop_total.to_csv('sid_pop_total', index = False)
uid_pop_train.to_csv('uid_pop_train', index = False)
sid_pop_train.to_csv('sid_pop_train', index = False)
