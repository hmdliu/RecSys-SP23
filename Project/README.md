# RecSys Final Project
## Abstract
Due to the natural long-tail distribution of user-item interactions, recommendation systems tend to favor popular items during prediction, resulting in pop- ularity bias. Previous work has demonstrated that enforcing a direct regularization on the BPR loss can significantly reduce the model bias while maintaining accuracy. However, it fails to achieve satisfactory performance for users with limited interaction histories. To alleviate this problem, this project proposes a systematic **mixed sampling strategy** to boost the debias performance without sacrificing the accuracy of recommendations, whose efficacy has been shown by the experiments on both synthetic and real-world datasets.

## Data
```shell
# PosInteraction uniform (epsilon = 0.0): ./data/final_synthetic3
# User uniform (epsilon = 1.0): ./data/final_synthetic1
# Mixed (epsilon = 0.2): ./data/final_synthetic0

# pre-process the synthetic dataset
cd [dataset_dir]
python step1.py
python step2.py

# pre-process the MovieLens-1M dataset
python step1.py
python step2.py
python step3.py [epsilon]
```

## Training
```shell
# [X] = synthetic0, synthetic1, synthetic3
# [Y] = none, pos2neg2, posneg
# [epsilon] = 0.0, 0.2, 1.0

# train on the synthetic dataset
python main_basic_synthetic.py --dataset [X] --model MF --sample [Y] --weight 0.8

# train on the movielens dataset (cmd)
python main_basic.py --model MF --dataset movielens --sample [Y] --weight 0.9 --reg yes --epsilon [epsilon]
# train on the movielens dataset (slurm)
sbatch scripts/movielens.slurm
```

## Misc
```shell
# visualizations: ./plot.ipynb

# dump training results (movielens)
python grep.py
```

## Group Members
- Haoming Liu (hl3797@nyu.edu)
- Haohai Pang (hp1397@nyu.edu)

## Acknowledgements
We thank Professor Hongyi Wen for his suggestions on the project. This work was supported through the NYU IT High Performance Computing resources, services, and staff expertise.
