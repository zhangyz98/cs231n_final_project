from __future__ import print_function
import zipfile
import os
import pdb
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import copy

#### (begin) separated fine and coarse / Yizhi 0521
class TransposeDepthInput_coarse(object):
    def __call__(self, depth):
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth)
        depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
        depth = nn.functional.interpolate(depth, size = (55, 74), mode='bilinear', align_corners=False)
        depth = torch.log(depth)
        # depth = (depth - depth.min())/(depth.max() - depth.min())
        return depth[0]
    
class TransposeDepthInput_fine(object):
    def __call__(self, depth):
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth)
        depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
        # depth = nn.functional.interpolate(depth, size = (55, 74), mode='bilinear', align_corners=False)
        # size * 2: trying up-sampling structure / Yizhi 0521
        depth = nn.functional.interpolate(depth, size = (55 * 2, 74 * 2), mode='bilinear', align_corners=False)
        depth = torch.log(depth)
        # depth = (depth - depth.min())/(depth.max() - depth.min())
        return depth[0]



rgb_data_transforms = transforms.Compose([
    transforms.Resize((228, 304)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

#### (end) separated fine and coarse / Yizhi 0521

'''
class TransposeDepthInput(object):
    def __call__(self, depth):
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth)
        depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
        # depth = nn.functional.interpolate(depth, size = (55, 74), mode='bilinear', align_corners=False)
        # size * 2: trying up-sampling structure / Yizhi 0521
        depth = nn.functional.interpolate(depth, size = (55 * 2, 74 * 2), mode='bilinear', align_corners=False)
        depth = torch.log(depth)
        # depth = (depth - depth.min())/(depth.max() - depth.min())
        return depth[0]

depth_data_transforms = transforms.Compose([
    TransposeDepthInput(),
])

'''

depth_data_transforms_fine = transforms.Compose([
    TransposeDepthInput_fine(),
])

depth_data_transforms_coarse = transforms.Compose([
    TransposeDepthInput_coarse(),
])

input_for_plot_transforms = transforms.Compose([
    transforms.Resize((55, 74)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

# Yizhi (0520)
def matrixflip(x):
    myl = np.array(x)
    return np.flip(myl, axis=0)
'''
flip_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomApply(torch.nn.ModuleList([
        color_jitter,
    ]), p=1),
    transforms.RandomGrayscale(p=1),
    transforms.ToTensor()
])
'''

'''
flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(1)
    # transforms.ToTensor()
])

color_transform = transforms.Compose([
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ]), p=1)
    # transforms.ToTensor()
])

def aug_transforms(image, depth, prob):
    d = torch.rand(1)
    if d < prob:
        flip_transform(image)
        # flip_transform(depth)
        matrixflip(depth)
    if prob < d and d < prob * 2:
        color_transform(image)
    return image, depth
'''

class NYUDataset(Dataset):
    def calculate_mean(self, images):
        mean_image = np.mean(images, axis=0)
        return mean_image

    def __init__(self, filename, type, rgb_transform = None, depth_transform = None):
        f = h5py.File(filename, 'r')
        # images_data = copy.deepcopy(f['images'][0:1449])
        # depths_data = copy.deepcopy(f['depths'][0:1449])
        # merged_data = np.concatenate((images_data, depths_data.reshape((1449, 1, 640, 480))), axis=1)

        # np.random.shuffle(merged_data)
        # images_data = merged_data[:,0:3,:,:]
        # depths_data = merged_data[:,3:4,:,:]

        # images_data = f['images'][0:1449] # commented by Yizhi (0520)
        # depths_data = f['depths'][0:1449]

        if type == "training":
            # self.images = images_data[0:1024]
            # self.depths = depths_data[0:1024]
            self.images = f['images'][0:1024] # images_data[0:1024]
            self.depths = f['depths'][0:1024] # depths_data[0:1024]
        elif type == "validation":
            self.images = f['images'][1024:1248] # images_data[1024:1248]
            self.depths = f['depths'][1024:1248] # depths_data[1024:1248]
            # self.images = images_data[1024:1072]
            # self.depths = depths_data[1024:1072]
        elif type == "test":
            self.images = f['images'][1248:] # images_data[1248:]
            self.depths = f['depths'][1248:] # depths_data[1248:]
            # self.images = images_data[0:32]
            # self.depths = depths_data[0:32]
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        # self.mean_image = self.calculate_mean(images_data[0:1449])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # image = (image - self.mean_image)/np.std(image)
        image = image.transpose((2, 1, 0))
        # modified by Yizhi (0511)
        # original_image = image.copy()
        # image = (image - image.min())/(image.max() - image.min())
        # image = image * 255
        # image = image.astype('uint8')
        image = Image.fromarray(image)
        # print(image)
        if self.rgb_transform:
            image = self.rgb_transform(image)
        
        depth = self.depths[idx]
        # modified by Yizhi (0511)
        # original_depth = depth.copy()
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((2, 1, 0))
        # print('depth:', depth.shape)
        if self.depth_transform:
            depth = self.depth_transform(depth)
        # if type == "validation": modified by Yizhi 
        # sample = {'image': image, 'depth': depth, 'original_image': original_image, 'original_depth': original_depth}
        # else:
        sample = {'image': image, 'depth': depth}
        
        return sample

# Modified by Yizhi (0520)
class NYUDataset_train(Dataset):
    def calculate_mean(self, images):
        mean_image = np.mean(images, axis=0)
        return mean_image

    def __init__(self, filename, img_folder, dep_folder, type, rgb_transform = None, depth_transform = None):
        f = h5py.File(filename, 'r')
        # images_data = copy.deepcopy(f['images'][0:1449])
        # depths_data = copy.deepcopy(f['depths'][0:1449])
        # merged_data = np.concatenate((images_data, depths_data.reshape((1449, 1, 640, 480))), axis=1)

        # np.random.shuffle(merged_data)
        # images_data = merged_data[:,0:3,:,:]
        # depths_data = merged_data[:,3:4,:,:]
        '''
        images_data = f['images'][0:1449]
        depths_data = f['depths'][0:1449]

        if type == "training":
            # self.images = images_data[0:1024]
            # self.depths = depths_data[0:1024]

            self.images = images_data[0:1024]
            self.depths = depths_data[0:1024]
        elif type == "validation":
            self.images = images_data[1024:1248]
            self.depths = depths_data[1024:1248]
            # self.images = images_data[1024:1072]
            # self.depths = depths_data[1024:1072]
        elif type == "test":
            self.images = images_data[1248:]
            self.depths = depths_data[1248:]
            # self.images = images_data[0:32]
            # self.depths = depths_data[0:32]
        '''
        self.depths = f['depths'][0:1024]
        self.img_path = img_folder
        self.dep_path = dep_folder
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        # self.mean_image = self.calculate_mean(images_data[0:1449])

    def __len__(self):
        return 1024 * 3 # len(self.images)

    def __getitem__(self, idx):
        # image = np.load(self.img_path+str(idx)+'.npy')
        # image = image.transpose((2, 1, 0))
        # image = Image.fromarray(image)
        image = Image.open(self.img_path+str(idx)+'.jpg')
        # print(image)
        if self.rgb_transform:
            image = self.rgb_transform(image)
        
        # depth = np.load(self.dep_path+str(idx)+'.npy')
        depth = self.depths[idx//3]
        if idx % 3 == 1:
            depth = np.flipud(depth).copy()
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((2, 1, 0))
        # print('depth:', depth.shape)
        if self.depth_transform:
            depth = self.depth_transform(depth)
            
        sample = {'image': image, 'depth': depth}
        
        return sample
    
        '''
        image = image.transpose((2, 1, 0))
        depth = self.depths[idx]
        depth = depth.T # Yizhi (0520)
        
        # Yizhi (0520)
        # print('before:', image.shape, depth.shape)
        image, depth = aug_transforms(Image.fromarray(image), depth, 0.1) 
        # print('after:', image, depth.shape)
        # image = Image.fromarray(image)
        
        # image = (image - self.mean_image)/np.std(image)
        # original_image = image.copy() # modified by Yizhi (0511)
        
        # image = (image - image.min())/(image.max() - image.min())
        # image = image * 255
        # image = image.astype('uint8')
        
        
        # depth = self.depths[idx]
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((1, 2, 0)) # ((2, 1, 0)) Yizhi 0520
        # modified by Yizhi (0511)
        # original_depth = depth.copy()
        
        if self.rgb_transform:
            image = self.rgb_transform(image)        
        if self.depth_transform:
            depth = self.depth_transform(depth)
        # if type == "validation": modified by Yizhi 
        # sample = {'image': image, 'depth': depth, 'original_image': original_image, 'original_depth': original_depth}
        # else:
        sample = {'image': image, 'depth': depth}
        '''
        
        return sample
