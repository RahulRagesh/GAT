# Graph Attention Networks

This is an optimized unofficial TensorFlow implementation of Graph Attention Networks. 

Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (ICLR 2018)

Optuna (https://optuna.org/) is used for hyperparameter search. 

This repository is heavily borrowed from https://github.com/tkipf/gcn. 
 

## Requirements
* tensorflow (Tested on 1.15.0)
* numpy
* scipy
* optuna
* networkx

## Run the demo
```bash
python train.py cora
```
