
import numpy as np
import pandas as pd
from similarity import Similarity

# turn off warnings for chained assignment
pd.options.mode.chained_assignment = None

class MemoryRecommender:
    """Implement the required functions for Q2"""
    def __init__(self, sim, sim_func, weight_type="item-based"):
        #TODO: implement any necessary initalization function here
        # sim_func is one of the similarity functions implemented in similarity.py 
        # weight_type is one of ["item-based", "user-based"]
        # You can add more input parameters as needed.
        assert weight_type in ('item-based', 'user-based')
        self.sim = sim
        self.sim_func = sim_func
        self.weight_type = weight_type

    def rating_predict(self, userID, itemID, verbose=True):
        #TODO: implement the rating prediction function for a given user-item pair
        if self.weight_type == 'item-based':
            item_set, df = self.sim.item_set(user=userID, return_df=True)
            if itemID in item_set:
                item_set.discard(itemID)
                df = df.drop(df['MovieID'] == itemID)
            df['Sim'] = df['MovieID'].apply(lambda j: self.sim_func(itemID, j, verbose=False))
        else:
            raise NotImplementedError
            user_set, df = self.sim.user_set(item=itemID, return_df=True)
            if userID in user_set:
                user_set.discard(userID)
                df = df.drop(df['UserID'] == userID)
            df['Sim'] = df['UserID'].apply(lambda v: self.sim_func(userID, v, verbose=False))
        r_pred = np.dot(df['Rating'].values, df['Sim'].values) / df['Sim'].values.sum()
        if verbose:
            print(f'\nr(user {userID}, item {itemID}) = {r_pred:.2f}')
        return r_pred
    
    def topk(self, userID, k=5):
        #TODO: implement top-k recommendations for a given user
        df = self.sim.dataset.movies_df
        rec_set = set(df['MovieID'].values) - self.sim.item_set(user=userID)
        rec_list = [(j, self.rating_predict(userID, j)) for j in list(rec_set)[:10]]    # test-only
        rec_list.sort(key=lambda x: x[1], reverse=True)
        for i in range(k):
            print(f'\nRank {i+1}: item {rec_list[i][0]} (r_pred = {rec_list[i][1]:.2f})')        

if __name__ == '__main__':
    
    sim = Similarity()
    # print the solution to Q3a here
    recommender = MemoryRecommender(sim, sim.cosine_similarity, weight_type="item-based")
    recommender.topk(userID=381)
    
    recommender = MemoryRecommender(sim, sim.pearson_similarity, weight_type="item-based")
    recommender.topk(userID=381)
    
    # print the solution to Q3b here
    recommender = MemoryRecommender(sim, sim.cosine_similarity, weight_type="user-based")
    recommender.topk(userID=381)