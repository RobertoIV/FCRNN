# FCRNN

Monocular depth estimation with FCRN from [Deeper Depth Prediction with Fully Convolutional Residual Networks](https://arxiv.org/abs/1606.00373) with internal surface normal constraint. Note this project is experimental and **not** carefully organized.

# Requirement

* Python 3.6
* PyTorch 0.4.0
* TensorboardX
* Other common packages like numpy, skimage


# Training

Download [NYU\_DEPTH v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [train/test splits matrix](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat) into `data`. Run `python run.py`.

# Demo

Download pretrained model from [here](https://pan.baidu.com/s/1AIbwgCJa_Dna9MJwPkFAuA) and put it in `checkpoints/normal_internal`. Then run `demo.ipynb` using jupyter.

