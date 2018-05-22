# Fairness in Machine Learning

This project demonstrates how make fair machine learning models.

![Fair training](https://github.com/equialgo/fairness-in-ml/raw/master/images/training.gif)


## Notebooks

- `fairness-in-ml.ipynb`: keras & TensorFlow implementation of [Towards fairness in ML with adversarial networks](https://blog.godatadriven.com/fairness-in-ml).
- `fairness-in-torch.ipynb`: PyTorch implementation of [Fairness in Machine Learning with PyTorch](http://blog.godatadriven.com/fairness-in-pytorch).
- `playground/*`: Various experiments.


## Getting started

This repo uses conda's virtual environment for __Python 3__.

Install (mini)conda if not yet installed.

For MacOS:
```sh
$ wget http://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh -O miniconda.sh
$ chmod +x miniconda.sh
$ ./miniconda.sh -b
```

`cd` into this directory and create the  conda virtual environment for Python 3 from `environment.yml`:
```sh
$ conda env create -f environment.yml
```

Activate the virtual environment:
```sh
$ source activate fairness-in-ml
```


## Contributing

If you have applied these models to a different dataset or implemented any other fair models, consider submitting a Pull Request!

