
import os
import sys
import pickle
import random
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
from collections import defaultdict

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn

EPOCHS = 20
NUM_NEG = 3
EPSILON = float(sys.argv[1])

def set_seed(seed=42):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_all_custom(test_num=100, dataset=None):
	""" We load all the three file here to save time in each epoch. """
    
	total_data = pd.read_csv('./total_df')    
	total_data = total_data[['uid', 'sid']]    
	total_data['uid'] = total_data['uid'].apply(lambda x : int(x))
	total_data['sid'] = total_data['sid'].apply(lambda x : int(x))    
	user_num = total_data['uid'].max() + 1
	item_num = total_data['sid'].max() + 1
	item_set = set(total_data['sid'].unique())
	user_set = set(total_data['uid'].unique())
	del total_data
    
	train_data = pd.read_csv('./train_df')    
	train_data = train_data[['uid', 'sid']]
	train_data['uid'] = train_data['uid'].apply(lambda x : int(x))
	train_data['sid'] = train_data['sid'].apply(lambda x : int(x))    
	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	test_data = pd.read_csv('./test_df')     
	test_data = test_data[['uid', 'sid']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))
	test_data.columns = ['user', 'item']    
	test_data = test_data.values.tolist()    
    
	val_data = pd.read_csv('./val_df')     
	val_data = val_data[['uid', 'sid']]
	val_data['uid'] = val_data['uid'].apply(lambda x : int(x))
	val_data['sid'] = val_data['sid'].apply(lambda x : int(x))
	val_data.columns = ['user', 'item']    
	val_data = val_data.values.tolist()        

	neg_samples_data = pd.read_csv('./neg_sample_df')     
	neg_samples_data = neg_samples_data[['uid', 'sid']]
	neg_samples_data['uid'] = neg_samples_data['uid'].apply(lambda x : int(x))
	neg_samples_data['sid'] = neg_samples_data['sid'].apply(lambda x : int(x))
	neg_samples_data.columns = ['user', 'item']    
	neg_samples_data = neg_samples_data.values.tolist()            

	'''
	test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in test_data:
		test_mat[x[0], x[1]] = 1.0
	'''
        
	total_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		total_mat[x[0], x[1]] = 1.0
	for x in test_data:
		total_mat[x[0], x[1]] = 1.0
	for x in val_data:
		total_mat[x[0], x[1]] = 1.0
	for x in neg_samples_data:
		total_mat[x[0], x[1]] = 1.0
        
	test_data = None # dummy code
	test_mat = None  # dummy code

	return train_data, test_data, user_num, item_num, train_mat, test_mat, total_mat, user_set, item_set

class BPRData(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, total_mat=None, num_ng=0, is_training=None, epsilon=EPSILON, user_set=set(), item_set=set()):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.features2 = None
		self.num_item = num_item
		self.train_mat = train_mat
		self.total_mat = total_mat
		self.num_ng = num_ng
		self.is_training = is_training       
		
		assert 0 <= epsilon <= 1
		self.epsilon = epsilon
		self.user_lst = list(user_set)
		self.item_set = item_set

		self.feat_num = len(features)
		self.pos_dict = defaultdict(set)
		self.neg_dict = defaultdict(set)
		for u, i in map(tuple, features):
			self.pos_dict[u].add(i)
		for u in self.pos_dict:
			self.neg_dict[u] = list(self.item_set - self.pos_dict[u])
		for k, v in self.pos_dict.items():
			self.pos_dict[k] = list(v)
            
	def __len__(self):
		return self.num_ng * len(self.features) if self.is_training else len(self.features)

	def __getitem__(self, idx):
		if random.random() > self.epsilon:
			u = random.choice(self.features)[0]     # interaction-uniform sampling
		else:
			u = random.choice(self.user_lst)        # user-uniform sampling
		pos1 = random.choice(self.pos_dict[u])
		pos2 = random.choice(self.pos_dict[u])
		neg1 = random.choice(self.neg_dict[u])
		neg2 = random.choice(self.neg_dict[u])
		return [u, pos1, pos2, neg1, neg2]

	def fetch_data(self):
		return [self[i] for i in range(len(self))]

if __name__ == '__main__':

	set_seed(0)
	print('using epsilon =', EPSILON)

	train_data, test_data, user_num, item_num, train_mat, test_mat, total_mat, user_set, item_set = load_all_custom()
	print('original user-pos tuple is', train_data[0:5])

	train_dataset = BPRData(train_data, item_num, train_mat, total_mat, num_ng=NUM_NEG, is_training=True, user_set=user_set, item_set=item_set)
	print('new (user, pos1, pos2, neg1, neg2) tuple is', train_dataset[0:5])

	root = f'./train_samples_{EPSILON:.1f}'
	os.makedirs(root)
	for i in range(EPOCHS):
		print('epoch', i)
		train_list = train_dataset.fetch_data()
		with open(f'{root}/train_samples_{i}', 'wb') as fp:
			pickle.dump(train_list, fp)
