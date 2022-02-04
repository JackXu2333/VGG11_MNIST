# VGG-11 Architecture Tested on MNIST

### Introduction

This repository contains PyTorch bottom up implementation of VGG-11 model
on pyTorch MNIST dataset with the following architecture.

```
- Conv(001, 064, 3, 1, 1) - BatchNorm(064) - ReLU - MaxPool(2, 2) 
- Conv(064, 128, 3, 1, 1) - BatchNorm(128) - ReLU - MaxPool(2, 2) 
- Conv(128, 256, 3, 1, 1) - BatchNorm(256) - ReLU 
- Conv(256, 256, 3, 1, 1) - BatchNorm(256) - ReLU - MaxPool(2, 2) 
- Conv(256, 512, 3, 1, 1) - BatchNorm(512) - ReLU 
- Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2) 
- Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU 
- Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2) 
- FC(0512, 4096) - ReLU - Dropout(0.5) 
- FC(4096, 4096) - ReLU - Dropout(0.5) 
- FC(4096, 10)
```

### Prerequisites

The models are implemented with config:
```
Python==3.8.3
torch==1.9.1
torchvision==0.10.1
matplotlib==3.4.3
numpy==1.21.2
```
Note that the test file automatically enable CUDA when available.

### MNIST dataset

The [MNIST](https://pytorch.org/vision/stable/datasets.html#mnist) dataset 
are used in training and testing of VGG-11.

### Training

#### Vanilla model

You can find the detail of training and testing the VGG-11 model 
on default MNIST dataset via [model_training.ipynb](https://github.com/JackXu2333/VGG11_MNIST/blob/master/model_training.ipynb)

#### Data Augmented model

You can find the detail of training and testing the VGG-11 model 
on default MNIST dataset via [model_training_augmented.ipynb](https://github.com/JackXu2333/VGG11_MNIST/blob/master/model_training_augmented.ipynb)

### Acknowledgment

If you find the model from this repository helpful, please cite the original [paper](https://arxiv.org/pdf/1409.1556.pdf):

```
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```
