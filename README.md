# Graph Attention Networks

This is an optimized unofficial TensorFlow implementation of Graph Attention Networks. 

Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)

Optuna (https://optuna.org/) is used for hyperparameter search. The hyperparameters set using ```trial.suggest*``` in the ```__call__``` method of ```TuneGAT``` class are searched. To run for a single configuration, all the hyperparameters have to be manually specified while setting ```n_trials=1```. 

## Extras
* There is a commented out code snippet to retrive the attention weights. 
* The code is also setup to support alternate optimization of attention weights and model weights. 
* On successful execution, a csv file with all the configurations and corresponding metrics will be dumped.

This repository is heavily borrowed from https://github.com/tkipf/gcn. 
 
## Requirements
* tensorflow (Tested on 1.15.0)
* numpy
* scipy
* optuna
* networkx

## To Run
```bash
python train.py cora
```

## Results 
* Number of heads [1st Layer] - 8 
* Number of heads [2nd Layer] - 1 
* Head Dimension - 8 
* Learning Rate, L2 Weight Decay, Model Dropout and Attention Dropout were searched through 500 trials.
* The model with the best Validation Accuracy is picked. Conflicts (if any) are resolved using the Test Accuracy.

| Dataset  | Train Accuracy | Val Accuracy | Test Accuracy | Best Test Accuracy |
|:--------:|:--------------:|:------------:|:-------------:|:------------------:|
|   Cora   |    0.9714      |    0.8300    |    0.8210     |       0.836        |
| Citeseer |    0.9167      |    0.7480    |    0.7300     |       0.744        |
|  Pubmed  |    0.9667      |    0.8240    |    0.7880     |       0.797        |
