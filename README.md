# Multi-information Fusion Graph Convolutional Network for Cancer Driver Gene Identiffcation

## Requirements

Our project is developed using PyTorch 1.7.1 in Python 3.7. 
We recommend you to use [anaconda](https://www.anaconda.com/) for dependency configuration.

First create an anaconda environment called ```MFGCN``` by

```shell
conda create -n MFGCN python=3.7
conda activate MFGCN
```

Then, you need to install torch manually to fit in with your server environment (e.g. CUDA version 11.0). Run:

```shell
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0
```

Besides, torch-scatter and torch-sparse are required for dealing with sparse graph, which can be installed using the following commands: 
```shell
pip install torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
pip install torch-geometric==1.6.3
```

## Result
The output of our model is supplied at './code/result/', which consist with two variates 'logits' and 'acc'.
The probability of each gene is non-cancer driver gene (the first column) or cancer driver gene (the second column) have been stored in 'logits', and the mean accuracy of 5-CV are in 'acc'
