# pytorch-syncbn

Tamaki Kojima(tamakoji@gmail.com)

## Announcement

**Pytorch 1.0 support**

## Overview
This is alternative implementation of "Synchronized Multi-GPU Batch Normalization" which computes global stats across gpus instead of locally computed. SyncBN are getting important for those input image is large, and must use multi-gpu to increase the minibatch-size for the training.

The code was inspired by [Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) and [Inplace-ABN](https://github.com/mapillary/inplace_abn)

## Remarks
- Unlike [Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), you don't need custom `nn.DataParallel`
- Unlike [Inplace-ABN](https://github.com/mapillary/inplace_abn), you can just replace your `nn.BatchNorm2d` to this module implementation, since it will not mark for inplace operation
- You can plug into arbitrary module written in PyTorch to enable Synchronized BatchNorm
- Backward computation is rewritten and tested against behavior of `nn.BatchNorm2d`

## Requirements
For PyTorch, please refer to https://pytorch.org/

NOTE : The code is tested only with PyTorch v1.0.0, CUDA10/CuDNN7.4.2 on ubuntu18.04

It utilize Pytorch JIT mechanism to compile seamlessly, using ninja. Please install ninja-build before use.

```
sudo apt-get install ninja-build
```

Also install all dependencies for python. For pip, run:


```
pip install -U -r requirements.txt
```

## Build

There is no need to build. just run and JIT will take care.
JIT and cpp extensions are supported after PyTorch0.4, however it is highly recommended to use PyTorch > 1.0 due to huge design changes.

## Usage

Please refer to [`test.py`](./test.py) for testing the difference between `nn.BatchNorm2d` and `modules.nn.BatchNorm2d`

```
import torch
from modules import nn as NN
num_gpu = torch.cuda.device_count()
model = nn.Sequential(
    nn.Conv2d(3, 3, 1, 1, bias=False),
    NN.BatchNorm2d(3),
    nn.ReLU(inplace=True),
    nn.Conv2d(3, 3, 1, 1, bias=False),
    NN.BatchNorm2d(3),
).cuda()
model = nn.DataParallel(model, device_ids=range(num_gpu))
x = torch.rand(num_gpu, 3, 2, 2).cuda()
z = model(x)
```

## Math

### Forward
1. compute <img src="https://latex.codecogs.com/gif.latex?\sum{x_i},\sum{x_i^2}"/> in each gpu
2. gather all <img src="https://latex.codecogs.com/gif.latex?\sum{x_i},\sum{x_i^2}"/> from workers to master and compute <img src="https://latex.codecogs.com/gif.latex?\mu,\sigma"/> where

    <img src="https://latex.codecogs.com/gif.latex?\mu=\frac{\sum{x_i}}{N}"/>

    and

    <img src="https://latex.codecogs.com/gif.latex?\sigma^2=\frac{\sum{x_i^2}-\mu\sum{x_i}}{N}"/></a>

    and then above global stats to be shared to all gpus, update running_mean and running_var by moving average using global stats.

3. forward batchnorm using global stats by

    <img src="https://latex.codecogs.com/gif.latex?\hat{x_i}=\frac{x_i-\mu}{\sqrt{\sigma^2&plus;\epsilon}}"/>

    and then

    <img src="https://latex.codecogs.com/gif.latex?y_i=\gamma\cdot\hat{x_i}&plus;\beta"/>

    where <img src="https://latex.codecogs.com/gif.latex?\gamma"/> is weight parameter and <img src="https://latex.codecogs.com/gif.latex?\beta"/> is bias parameter.

4. save <img src="https://latex.codecogs.com/gif.latex?x,&space;\gamma\&space;\beta,&space;\mu,&space;\sigma^2"/> for backward

### Backward

1. Restore saved <img src="https://latex.codecogs.com/gif.latex?x,&space;\gamma\&space;\beta,&space;\mu,&space;\sigma^2"/>

2. Compute below sums on each gpu

    <img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{N_j}(\frac{dJ}{dy_i})"/>

    and

    <img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{N_j}(\frac{dJ}{dy_i}\cdot\hat{x_i})"/>

    where <img src="https://latex.codecogs.com/gif.latex?j\in[0,1,....,num\_gpu]"/>

    then gather them at master node to sum up global, and normalize with N where N is total number of elements for each channels. Global sums are then shared among all gpus.

3. compute gradients using global stats

    <img src="https://latex.codecogs.com/gif.latex?\frac{dJ}{dx_i},&space;\frac{dJ}{d\gamma},&space;\frac{dJ}{d\beta}&space;"/>

    where

    <img src="https://latex.codecogs.com/gif.latex?\frac{dJ}{d\gamma}=\sum_{i=1}^{N}(\frac{dJ}{dy_i}\cdot\hat{x_i})"/>

    and

    <img src="https://latex.codecogs.com/gif.latex?\frac{dJ}{d\beta}=\sum_{i=1}^{N}(\frac{dJ}{dy_i})"/>

    and finally,

    <img src="https://latex.codecogs.com/gif.latex?\frac{dJ}{dx_i}=\frac{dJ}{d\hat{x_i}}\frac{d\hat{x_i}}{dx_i}+\frac{dJ}{d\mu_i}\frac{d\mu_i}{dx_i}+\frac{dJ}{d\sigma^2_i}\frac{d\sigma^2_i}{dx_i}"/>  

    <img src="https://latex.codecogs.com/gif.latex?=\frac{1}{N\sqrt{(\sigma^2+\epsilon)}}(N\frac{dJ}{d\hat{x_i}}-\sum_{j=1}^{N}(\frac{dJ}{d\hat{x_j}})-\hat{x_i}\sum_{j=1}^{N}(\frac{dJ}{d\hat{x_j}}\hat{x_j}))"/>  

    <img src="https://latex.codecogs.com/gif.latex?=\frac{\gamma}{N\sqrt{(\sigma^2+\epsilon)}}(N\frac{dJ}{dy_i}-\sum_{j=1}^{N}(\frac{dJ}{dy_j})-\hat{x_i}\sum_{j=1}^{N}(\frac{dJ}{dy_j}\hat{x_j}))"/>  

   Note that in the implementation, normalization with N is performed at step (2) and above equation and implementation is not exactly the same, but mathematically is same.

   You can go deeper on above explanation at [Kevin Zakka's Blog](https://kevinzakka.github.io/2016/09/14/batch_normalization/)