
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

SAVE_ROOT = './CA1'
DATASET_ROOT = './CA1/ml-1m'
USERS_COLUMN = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
MOVIES_COLUMN = ['MovieID', 'Title', 'Genres']
RATINGS_COLUMN = ['UserID', 'MovieID', 'Rating', 'Timestamp']
AGE_GROUP_TICKS = ['<18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+']

class RecDataset:
    """Implement the required functions for Q1"""
    def __init__(self, root=DATASET_ROOT):
        #TODO: implement any necessary initalization function such as data loading here
        #You can add more input parameters as needed.
        
        # init dataset path & check validity
        self.root = root
        self.users_path = os.path.join(root, 'users.dat')
        self.movies_path = os.path.join(root, 'movies.dat')
        self.ratings_path = os.path.join(root, 'ratings.dat')
        assert os.path.exists(self.users_path)
        assert os.path.exists(self.movies_path)
        assert os.path.exists(self.ratings_path)

        # load users
        print(f'\nLoading users data from [{self.users_path}]:')
        self.users_df = pd.read_csv(self.users_path, header=None, sep='::', names=USERS_COLUMN, engine='python')
        print(self.users_df.head())

        # load movies
        print(f'\nLoading movies data from [{self.movies_path}]:')
        self.movies_df = pd.read_csv(self.movies_path, header=None, sep='::', names=MOVIES_COLUMN, engine='python')
        print(self.movies_df.head())

        # load ratings
        print(f'\nLoading ratings data from [{self.ratings_path}]:')
        self.ratings_df = pd.read_csv(self.ratings_path, header=None, sep='::', names=RATINGS_COLUMN, engine='python')
        print(self.ratings_df.head())

        # parse dataset statistics (for Q1a)
        self.nb_users = self.users_df.shape[0]
        self.nb_items = self.movies_df.shape[0]
        self.nb_ratings = self.ratings_df.shape[0]
        self.ratings_count = self.ratings_df['UserID'].value_counts()
        self.ratings_max = max(self.ratings_count)
        self.ratings_min = min(self.ratings_count)

    def describe(self):
        #TODO: implement the required functions and print the solution to Question 1a here
        print(f'\n[Dataset Info]')
        print(f'Root path: {self.root}')
        print(f'Num of unique users: {self.nb_users}')
        print(f'Num of unique items: {self.nb_items}')
        print(f'Num of ratings: {self.nb_ratings}')
        print(f'Max rated items has {self.ratings_max} ratings')
        print(f'Min rated items has {self.ratings_min} ratings')
        
    def query_user(self, userID, verbose=True):
        #TODO: implement the required functions and print the solution to Question 1b here
        assert userID in self.users_df['UserID']
        user_df = self.ratings_df[self.ratings_df['UserID'] == userID]
        ratings_num = user_df.shape[0]
        ratings_avg = user_df['Rating'].mean()
        if verbose:
            print(f'\n[User {userID} Info]')
            print(f'Num of ratings: {ratings_num}')
            print(f'Averaged ratings score: {ratings_avg:.2f}')
        return ratings_num, ratings_avg
    
    def dist_by_age_groups(self):
        #TODO: implement the required functions and print the solution to Question 1c here
        #You could import `users.dat` here or in __init__(). 
        #This function is expected to return two lists - you shall use these lists to 
        #draw the bar plots and attach them in your answer sheet.
        
        # group users by age groups
        group_dict = defaultdict(list)
        for userID, group in self.ratings_df.groupby('UserID'):
            group_dict[self.users_df.iloc[userID-1]['Age']].append(group)
        group_dict = {k:pd.concat(v) for k, v in group_dict.items()}

        # calculate distributions
        key_list = sorted(group_dict.keys())
        rating_num_list, rating_avg_list = [], []
        for k in key_list:
            rating_num_list.append(group_dict[k].shape[0])
            rating_avg_list.append(group_dict[k]['Rating'].mean())

        # plot figures
        plt.figure(figsize=(10, 4), dpi=120)
        plt.subplot(1, 2, 1)
        plt.xlabel('Age Groups')
        plt.ylabel('Num of Ratings')
        plt.bar(AGE_GROUP_TICKS, rating_num_list)
        plt.subplot(1, 2, 2)
        plt.xlabel('Age Groups')
        plt.ylabel('Average Rating Score')
        plt.bar(AGE_GROUP_TICKS, rating_avg_list)
        plt.savefig(os.path.join(SAVE_ROOT, 'Q1c.png'))
        plt.show()

if __name__ == '__main__':
    
    dataset = RecDataset()
    
    # print the solution to Q1a here
    dataset.describe() 
    
    # print the solution to Q1b here
    dataset.query_user(userID=100)
    dataset.query_user(userID=381) 
    
    # print the solution to Q1c here
    dataset.dist_by_age_groups()