
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# init data
user_item_mtx = np.zeros((200, 200))
for i in range(200):
    for j in range(200):
        if (i < j):
            user_item_mtx[i,200-j-1] = 1            
        if (i > j):
            user_item_mtx[i,200-j-1] = 0
user_item_mtx[1, 200 - 1] = 1
user_item_mtx[200-1, 0] = 1
print(user_item_mtx)

# visualization
heatmap = sns.heatmap(user_item_mtx)
fig = heatmap.get_figure()
fig.savefig("heatmap.png")

# dump data
data = pd.DataFrame(user_item_mtx)
data = data.unstack().reset_index()
data.columns = ['sid', 'uid', 'ratings']
pos_data = data[data['ratings'] == 1.0]
all_data = data[['uid', 'sid']]
train_df = pos_data[['uid', 'sid']].reset_index()[['uid', 'sid']]
test_df = train_df
val_df = train_df
pos_data.to_csv('total_df', index = False)
train_df.to_csv('train_df', index = False)
val_df.to_csv('val_df', index = False)
test_df.to_csv('test_df', index = False)

# dump popularity
uid_pop_total = pos_data.uid.value_counts().reset_index()
uid_pop_total.columns = ['uid', 'total_counts']
sid_pop_total = pos_data.sid.value_counts().reset_index()
sid_pop_total.columns = ['sid', 'total_counts']
uid_pop_train = train_df.uid.value_counts().reset_index()
uid_pop_train.columns = ['uid', 'train_counts']
sid_pop_train = train_df.sid.value_counts().reset_index()
sid_pop_train.columns = ['sid', 'train_counts']
uid_pop_total.to_csv('uid_pop_total', index = False)
sid_pop_total.to_csv('sid_pop_total', index = False)
uid_pop_train.to_csv('uid_pop_train', index = False)
sid_pop_train.to_csv('sid_pop_train', index = False)