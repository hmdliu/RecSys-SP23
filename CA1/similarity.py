
import numpy as np
from rec_dataset import RecDataset

class Similarity:
    """Implement the required functions for Q2"""
    def __init__(self):
        #TODO: implement any necessary initalization function here
        #You can add more input parameters as needed.
        self.dataset = RecDataset()
    
    # fetch the set of users who have been interacted with an item
    def user_set(self, item):
        df = self.dataset.ratings_df
        df = df[df['MovieID'] == item]
        return set(df['UserID'].values)

    def jaccard_similarity(self, item1, item2, verbose=True):
        #TODO: implement the required functions and print the solution to Question 2a here
        us1, us2 = self.user_set(item1), self.user_set(item2)
        jaccard_sim = len(us1 & us2) / len(us1 | us2)
        if verbose:
            print(f'\nJaccard similarity between item {item1} and item {item2} is {jaccard_sim:.2f}')
        return jaccard_sim
        
    def cosine_similarity(self, item1, item2, verbose=True):
        #TODO: implement the required functions and print the solution to Question 2b here
        df = self.dataset.ratings_df
        userID_set = self.user_set(item1) & self.user_set(item2)
        vec1 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item1)]['Rating'].values
        vec2 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item2)]['Rating'].values
        cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec1))
        if verbose:
            print(f'\nCosine similarity between item {item1} and item {item2} is {cosine_sim:.2f}')
        return cosine_sim

    def pearson_similarity(self, item1, item2, verbose=True):
        #TODO: implement the required functions and print the solution to Question 2c here
        df = self.dataset.ratings_df
        userID_set = self.user_set(item1) & self.user_set(item2)
        vec1 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item1)]['Rating'].values
        vec2 = df[(df['UserID'].isin(userID_set)) & (df['MovieID'] == item2)]['Rating'].values
        vec1 = vec1 - vec1.mean()
        vec2 = vec2 - vec2.mean()
        pearson_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec1))
        if verbose:
            print(f'\nPearson similarity between item {item1} and item {item2} is {pearson_sim:.2f}')
        return pearson_sim

if __name__ == '__main__':
    
    sim = Similarity()
    
    # print the solution to Q2a here
    sim.jaccard_similarity(item1=1, item2=2)
    sim.jaccard_similarity(item1=1, item2=3114)
    
    # print the solution to Q2b here
    sim.cosine_similarity(item1=1, item2=2)
    sim.cosine_similarity(item1=1, item2=3114)
    
    # print the solution to Q2c here
    sim.pearson_similarity(item1=1, item2=2)
    sim.pearson_similarity(item1=1, item2=3114)