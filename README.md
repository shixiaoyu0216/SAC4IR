# Soft Actor-Critic framework for Interactive Recommendation (SAC4IR)

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/shixiaoyu0216/SAC4IR/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.1+cu118-%237732a8)](https://pytorch.org/)

This repository contains the official Pytorch implementation for the paper *Towards Long-term Fairness in Interactive Recommendation: A Maximum Entropy Reinforcement Learning Approach* and *Maximum Entropy Policy for Long-term Fairness in Interactive Recommender Systems*. It also contains the a dataset of SAC4IR: *Movielens-1m*.

<img src="figs/Introduction.png" alt="Introduction" style="zoom:100%;" />

More descriptions are available via the [paper](https://www.computer.org/csdl/proceedings-article/icws/2023/048500a118/1QzhT2tdnGg), and a new paper under review.

If this work helps you, please kindly cite our papers:

```latex
@inproceedings{shi2023towards,
  title={Towards Long-term Fairness in Interactive Recommendation: A Maximum Entropy Reinforcement Learning Approach},
  author={Shi, Xiaoyu and Liu, Quanliang and Xie, Hong and Shang, Mingsheng},
  booktitle={2023 IEEE International Conference on Web Services (ICWS)},
  pages={118--123},
  year={2023},
  organization={IEEE}
}
```

---
## Installation

1. Clone this git repository and change directory to this repository:

	```bash
	git clone git@github.com:shixiaoyu0216/SAC4IR.git
	cd SAC4IR
	```


2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

    ```bash
    conda create --name SAC4IR python=3.9 -y
    ```

3. Activate the newly created environment.

    ```bash
    conda activate SAC4IR
    ```


4. Install the required 

    ```bash
    pip install -r requirements.txt
    ```


## Download the data

1. Download the compressed dataset

    ```bash 
    wget http://files.grouplens.org/datasets/movielens/ml-1m.zip
	unzip ml-1m.zip -d ml-1m
    ```

 	```bash 
    wget https://guoguibing.github.io/librec/datasets/CiaoDVD.zip
	unzip CiaoDVD.zip -d CiaoDVD
    ```

	```bash 
    You can download ratings only data from http://jmcauley.ucsd.edu/data/amazon/.
    ```

If you use them in your work, please cite: [![LINK](https://img.shields.io/badge/-Paper%20Link-lightgrey)](https://dl.acm.org/doi/abs/10.1145/2827872) [![PDF](https://img.shields.io/badge/-PDF-red)](https://dl.acm.org/doi/abs/10.1145/2827872), [![LINK](https://img.shields.io/badge/-Paper%20Link-lightgrey)](https://ieeexplore.ieee.org/abstract/document/6921639)[![PDF](https://img.shields.io/badge/-PDF-red)](https://ieeexplore.ieee.org/abstract/document/6921639), [![LINK](https://img.shields.io/badge/-Paper%20Link-lightgrey)](https://dl.acm.org/doi/abs/10.1145/2872427.2883037) [![PDF](https://img.shields.io/badge/-PDF-red)](https://dl.acm.org/doi/abs/10.1145/2872427.2883037)


```latex
@article{harper2015movielens,
  title={The movielens datasets: History and context},
  author={Harper, F Maxwell and Konstan, Joseph A},
  journal={Acm transactions on interactive intelligent systems (tiis)},
  volume={5},
  number={4},
  pages={1--19},
  year={2015},
  publisher={Acm New York, NY, USA}
}

@INPROCEEDINGS{guo2014etaf,
  author = {Guo, G. and Zhang, J. and Thalmann, D. and Yorke-Smith, N.},
  title = {ETAF: An Extended Trust Antecedents Framework for Trust Prediction},
  booktitle = {Proceedings of the 2014 International Conference on Advances in Social Networks Analysis and Mining (ASONAM)},
  pages = {540-547},
  year = {2014}
} 

@inproceedings{he2016ups,
  title={Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering},
  author={He, Ruining and McAuley, Julian},
  booktitle={proceedings of the 25th international conference on world wide web},
  pages={507--517},
  year={2016}
}
```

If things go well, you can run the following examples now.

---
## Examples to run the code

The following commands only give one example. For more details, please kindly refer to the paper and the definitions in the code files. 

- #### Movielens-1m

Train the the RL policy using Movielens-1m

    ```bash
    python main.py
    ```

---
## A guide to reproduce the main results of the paper.

You can follow the guide to reproduce the main results of baselines, see [ItemPop](https://recbole.io/docs/user_guide/model/general/pop.html), [BPRMF](https://recbole.io/docs/user_guide/model/general/bpr.html), [MF-IPS and DICE](https://github.com/JingsenZhang/Recbole-Debias), [C2UCB](https://github.com/YunSeo00/combinatorial_MAB), [FCPO](https://github.com/TobyGE/FCPO), [FairCo](https://github.com/MarcoMorik/Dynamic-Fairness).

The details of VirtualTaobao can be referred to [VirtualTaobao](https://github.com/eyounx/VirtualTaobao).
