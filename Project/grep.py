
import os
import pandas as pd

res_list = []
info_list = []
ratio_list = ['0.0', '0.1', '0.2', '1.0']
method_list = ['none', 'pos2neg2', 'posneg']
template = './experiments/MF_movielens_{}_20_0.9_burninno_regyes_{}.csv'

for m in method_list:
    for r in ratio_list:
        path = template.format(m, r)
        try:
            assert os.path.exists(path), path
        except:
            continue
        res_list.append(pd.read_csv(path).tail(1))
        info_list.append(f'{m}_{r}')

res_df = pd.concat(res_list)
res_df = res_df[['batch', 'sample', 'weight', 'HR', 'NDCG', 'SCC_rank', 'mean']]
res_df.columns = ['batch', 'sample', 'weight', 'HR', 'NDCG', 'iPO', 'PopQ@1']
res_df['sample'] = info_list
print(res_df)
