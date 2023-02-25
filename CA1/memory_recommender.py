
import os
import time
import pickle
import numpy as np
import pandas as pd

# acceleration trick (credits to Ruochen)
import multiprocessing as mp

from similarity import Similarity
from rec_dataset import SAVE_ROOT

# turn off warnings for chained assignment
pd.options.mode.chained_assignment = None

class MemoryRecommender:
    """Implement the required functions for Q2"""
    def __init__(self, sim, func_type="item_cosine", weight_type="item-based"):
        #TODO: implement any necessary initalization function here
        # sim_func is one of the similarity functions implemented in similarity.py 
        # weight_type is one of ["item-based", "user-based"]
        # You can add more input parameters as needed.
        assert weight_type in ('item-based', 'user-based')
        self.sim = sim
        self.func_type = func_type
        self.weight_type = weight_type
        self.sim_func = dict(
            item_jaccard=self.sim.item_jaccard_similarity,
            item_cosine=self.sim.item_cosine_similarity,
            user_cosine=self.sim.user_cosine_similarity,
            item_pearson=self.sim.item_pearson_similarity,
        )[func_type]
        os.makedirs(os.path.join(SAVE_ROOT, 'mem'), exist_ok=True)

    def rating_predict(self, userID, itemID, verbose=True):
        #TODO: implement the rating prediction function for a given user-item pair
        if self.weight_type == 'item-based':
            item_set, df = self.sim.item_set(user=userID, return_df=True)
            if itemID in item_set:
                item_set.discard(itemID)
                df = df.drop(df['MovieID'] == itemID)
            df['Sim'] = df['MovieID'].apply(lambda j: self.sim_func(itemID, j, verbose=False))
        else:
            user_set, df = self.sim.user_set(item=itemID, return_df=True)
            if userID in user_set:
                user_set.discard(userID)
                df = df.drop(df['UserID'] == userID)
            df['Sim'] = df['UserID'].apply(lambda v: self.sim_func(userID, v, verbose=False))
        r_pred = np.dot(df['Rating'].values, df['Sim'].values) / (df['Sim'].values.sum() + 1e-8)
        if verbose:
            print(f'\nr(user {userID}, item {itemID}) = {r_pred:.4f}')
        return r_pred, userID, itemID
    
    def dump_similarity(self, userID):
        start_time = time.time()
        print(f'\nComputing similarity matrix for user {userID}')
        # compute the similarity scores for this user via multi-processing
        df = self.sim.dataset.movies_df
        rec_set = set(df['MovieID'].values) - self.sim.item_set(user=userID)
        if (cores := mp.cpu_count()) > 1:
            print(f'\nMulti-processing with {cores} CPUs')
            with mp.Pool(cores) as p:
                rec_list = p.starmap(self.rating_predict, [(userID, j) for j in list(rec_set)])
        else:
            print(f'\nMulti-processing disabled')
            rec_list = [self.rating_predict(userID, j) for j in list(rec_set)]
        rec_list.sort(key=lambda x: x[0], reverse=True)
        # dump the sorted similarity list
        dump_path = os.path.join(SAVE_ROOT, f'mem/{self.func_type}_uid{userID}.pkl')
        print(f'\nDumping similarity matrix to [{dump_path}]...')
        with open(dump_path, 'wb') as f:
            pickle.dump(rec_list, f)
        print(f'\nComputation time: {(time.time() - start_time) / 60:.2f} mins')
        print('\n================================================================')

    def topk(self, userID, k=5):
        #TODO: implement top-k recommendations for a given user
        load_path = os.path.join(SAVE_ROOT, f'mem/{self.func_type}_uid{userID}.pkl')
        print(f'\nLoading similarity matrix from [{load_path}]...')
        with open(load_path, 'rb') as f:
            rec_list = pickle.load(f)
        print(f'\nTop-{k} recommendations for user {userID} ({self.func_type}):')
        for i in range(k):
            print(f'\nRank {i+1}: item {rec_list[i][2]} (r_pred = {rec_list[i][0]:.4f})')
        print('\n================================================================')
        

if __name__ == '__main__':
    
    sim = Similarity()

    # print the solution to Q3a here
    recommender = MemoryRecommender(sim, func_type="item_cosine", weight_type="item-based")
    recommender.dump_similarity(userID=381)
    recommender.topk(userID=381)
    
    recommender = MemoryRecommender(sim, func_type="item_pearson", weight_type="item-based")
    recommender.dump_similarity(userID=381)
    recommender.topk(userID=381)
    
    # print the solution to Q3b here
    recommender = MemoryRecommender(sim, func_type="user_cosine", weight_type="user-based")
    recommender.dump_similarity(userID=381)
    recommender.topk(userID=381)