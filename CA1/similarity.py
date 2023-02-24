
import numpy as np
from rec_dataset import RecDataset

COS_SIM = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

class Similarity:
    """Implement the required functions for Q2"""
    def __init__(self):
        #TODO: implement any necessary initalization function here
        #You can add more input parameters as needed.
        self.dataset = RecDataset()

    # fetch the set of users who have been interacted with an item
    def user_set(self, item, return_df=False):
        df = self.dataset.ratings_df
        df = df[df['MovieID'] == item]
        return (set(df['UserID'].values), df) if return_df else set(df['UserID'].values)
    
    # fetch the set of items who have been interacted with a user
    def item_set(self, user, return_df=False):
        df = self.dataset.ratings_df
        df = df[df['UserID'] == user]
        return (set(df['MovieID'].values), df) if return_df else set(df['MovieID'].values)

    def item_jaccard_similarity(self, item1, item2, verbose=True):
        #TODO: implement the required functions and print the solution to Question 2a here
        us1, us2 = self.user_set(item1), self.user_set(item2)
        sim = len(us1 & us2) / (len(us1 | us2) + 1e-8)
        if verbose:
            print(f'\nJaccard similarity between item {item1} and item {item2} is {sim:.4f}')
        return sim
        
    def item_cosine_similarity(self, item1, item2, verbose=True):
        #TODO: implement the required functions and print the solution to Question 2b here
        df = self.dataset.ratings_df
        userID_set = self.user_set(item1) & self.user_set(item2)
        df1 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item1)]
        df2 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item2)]
        sim = COS_SIM(df1['Rating'].values, df2['Rating'].values) if len(userID_set) > 0 else 0
        if verbose:
            print(f'\nCosine similarity between item {item1} and item {item2} is {sim:.4f}')
        return sim
    
    def user_cosine_similarity(self, user1, user2, verbose=True):
        #TODO: implement the required functions and print the solution to Question 3b here
        df = self.dataset.ratings_df
        movieID_set = self.item_set(user1) & self.item_set(user2)
        df1 = df[(df['MovieID'].isin(movieID_set)) & (df['UserID'] == user1)].sort_values('MovieID')
        df2 = df[(df['MovieID'].isin(movieID_set)) & (df['UserID'] == user2)].sort_values('MovieID')
        sim = COS_SIM(df1['Rating'].values, df2['Rating'].values) if len(movieID_set) > 0 else 0
        if verbose:
            print(f'\nCosine similarity between user {user1} and user {user2} is {sim:.4f}')
        return sim

    def item_pearson_similarity(self, item1, item2, verbose=True):
        #TODO: implement the required functions and print the solution to Question 2c here
        df = self.dataset.ratings_df
        userID_set = self.user_set(item1) & self.user_set(item2)
        vec1 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item1)]['Rating'].values
        vec2 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item2)]['Rating'].values
        sim = COS_SIM(vec1 - vec1.mean(), vec2 - vec2.mean()) if len(userID_set) > 0 else 0
        if verbose:
            print(f'\nPearson similarity between item {item1} and item {item2} is {sim:.4f}')
        return sim

if __name__ == '__main__':
    
    sim = Similarity()
    
    # print the solution to Q2a here
    sim.item_jaccard_similarity(item1=1, item2=2)
    sim.item_jaccard_similarity(item1=1, item2=3114)
    
    # print the solution to Q2b here
    sim.item_cosine_similarity(item1=1, item2=2)
    sim.item_cosine_similarity(item1=1, item2=3114)
    
    # print the solution to Q2c here
    sim.item_pearson_similarity(item1=1, item2=2)
    sim.item_pearson_similarity(item1=1, item2=3114)