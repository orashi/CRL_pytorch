import torch
import torch.utils.data as data
from torchvision import transforms
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import glob
import sys
import re
import random
import math
from PIL import Image
import numbers
from .utils.loader import pfm_read, color_read

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_pfm_file(filename):
    return filename.endswith('.pfm')


def make_ft_dataset(root, train=True):
    status = 'TRAIN' if train else 'TEST'
    stereo = os.path.join(root, 'frames_cleanpass', status)
    disparity = os.path.join(root, 'disparity', status)
    image_groups = []

    for type in ['A', 'B', 'C']:
        for scene in sorted(os.listdir(os.path.join(stereo, type))):
            for fname in sorted(os.listdir(os.path.join(stereo, type, scene, 'left'))):
                if is_image_file(fname):
                    Lpath, Rpath = os.path.join(stereo, type, scene, 'left', fname), os.path.join(stereo, type, scene,
                                                                                                  'right', fname)
                    LDpath = os.path.join(disparity, type, scene, 'left', fname[:-3]+'pfm')
                    image_groups.append((Lpath, Rpath, LDpath))

    return image_groups


def make_kitti_dataset(root, train):
    pass


class ImageFolder(data.Dataset):
    def __init__(self, root, indexer, transform=None, Dtransform=None):
        imgs = indexer(root)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in folders."))
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        Lpath, Rpath, LDpath = self.imgs[index]
        Limg, Rimg, Ldisparity = color_read(Lpath), color_read(Rpath), pfm_read(LDpath)

        # TODO: discuss this part
        # if random.random() < 0.5:
        #     Cimg, Simg = Cimg.transpose(Image.FLIP_TOP_BOTTOM), Simg.transpose(Image.FLIP_TOP_BOTTOM)
        Limg, Rimg, Ldisparity = self.transform(Limg), self.transform(Rimg), self.Dtransform(Ldisparity)

        return Limg, Rimg, Ldisparity

    def __len__(self):
        return len(self.imgs)


def CreateFT3DLoader(opt):
    random.seed(opt.manualSeed)

    CTrans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    DTrans = transforms.Lambda(lambda x: torch.from_numpy(x.transpose((2, 0, 1))).float())

    dataset = ImageFolder(root=opt.FT3D, indexer=make_ft_dataset, transform=CTrans, Dtransform=DTrans)
    assert dataset

    return data.DataLoader(dataset, batch_size=opt.batchSize,
                           shuffle=True, num_workers=int(opt.workers), drop_last=True)
