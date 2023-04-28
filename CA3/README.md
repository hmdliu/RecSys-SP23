# RecSys-SP23 Coding Assignment 3

## Dataset Preparation
```shell
# default path: ./ml-1m
bash scripts/prepare_dataset.sh

# data pre-processing
python -m src.dataset
```

## Training
**Split list**: `loo_all`, `loo_male`, `loo_female`, `hp_all`, `hp_male`,`hp_female`.
```shell
# using shell
python -m src.train_nmf [split]
python -m src.train_bpr [split]

# using slurm
sbatch scripts/train.slurm [split]

# dump results
python scripts/dump_results.py
```

## Testing
```shell
# using shell
python -m src.test_nmf [split]
python -m src.test_bpr [split]
```