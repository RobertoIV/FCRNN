import h5py
import numpy as np
import torch
import os
import scipy
from scipy import io
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('agg')
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import transform as trans
import skimage
import cv2
import random

"""
transforms:
- rescale to 240, 320
-
"""

class NYUDepth(Dataset):
    # rgb intrinsic parameters
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    
    def __init__(self, root, mode):
        assert mode in ['train', 'test'], 'invalid dataset mode'
        self.root = root
        self.mat_path = os.path.join(root, 'nyu_depth_v2_labeled.mat')
        self.split_path = os.path.join(root, 'splits.mat')
        splits = io.loadmat(self.split_path)
        self.mode = mode
        if mode == 'train':
            self.indices = splits['trainNdxs']
        else:
            self.indices = splits['testNdxs']
    
    def __getitem__(self, index):
        with h5py.File(self.mat_path) as f:
            images = f['images']
            depths = f['depths']
            # image (3, w, h), uint8
            # depth (w, h) float
            image = images[self.indices[index, 0]]
            depth = depths[self.indices[index, 0]]
            # transpose to (h, w, 3), (h, w)
            image = image.transpose(2, 1, 0)
            depth = depth.transpose(1, 0)
            # compute point cloud (h, w, 3)
            cloud = depth_to_points(depth, self.fx_rgb, self.fy_rgb, self.cx_rgb, self.cy_rgb)
            # to float
            image = skimage.img_as_float(image)
            # rescale
            image = trans.rescale(image, 0.5, mode='constant')
            depth = trans.rescale(depth, 0.5, mode='constant', preserve_range=True)
            cloud = trans.rescale(cloud, 0.5, mode='constant', preserve_range=True)
            # random crop
            image, depth, cloud = random_crop(image, depth, cloud, size=(228, 304))
            # to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            depth = torch.from_numpy(depth).float()
            cloud = torch.from_numpy(cloud).float()
            
            # depth to (1, H, W)
            depth = depth[None]
            
            return image, depth, cloud
    
    def __len__(self):
        return len(self.indices)


def depth_to_points(depths, fx, fy, cx, cy):
    """
    Reproject a depth map to point clouds.

    :param depths: (H, W)
    :param fx, fy, cx, cy: intrinsic parameters
    :return (H, W, 3), point cloud
    """
    H, W = depths.shape
    # make meshgrid (H, W), (H, W)
    x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H))
    # get 3-d point cloud
    X = (x_grid - cx) * depths / fx
    Y = (y_grid - cy) * depths / fy
    Z = depths.copy()
    # combine into (X, Y, Z)
    cloud = np.stack((X, Y, Z), axis=2)
    
    return cloud


def random_crop(*arrays, size):
    """
    Perform the same random crop to a set of arrays.

    :param arrays: each of shape (H, W, *)
    :param size: (h, w), size after cropping
    """
    h_old, w_old = arrays[0].shape[:2]
    h_new, w_new = size
    h_offset = random.randint(0, h_old - h_new)
    w_offset = random.randint(0, w_old - w_new)
    
    return [a[h_offset:h_offset + h_new, w_offset:w_offset + w_new] for a in arrays]


def show_point_cloud(cloud, color, savepth=None):
    """
    Visualize a point cloud

    :param cloud: (H, W, 3)
    """
    
    # plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    cloud = cloud.reshape(-1, 3)
    ax.scatter3D(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=color.reshape(-1, 3), s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(-80, -90)
    # plt.show()
    if savepth is not None:
        plt.savefig(savepth)
    
if __name__ == '__main__':
    path = 'data/NYU_DEPTH'
    a = NYUDepth(path, 'train')
    img, depth, cloud = a[1]
    show_point_cloud(cloud, img.permute(1, 2, 0))

