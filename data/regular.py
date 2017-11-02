import torch
import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
import glob
import sys
import re
import random
import math
import numbers
from .utils.loader import *


def load_dataset(dir, split=80, transform=None, target_transform=None, co_transform=None):
    train_dataset_list, test_dataset_list = make_dataset(dir, split)
    train_dataset = ListDataset(train_dataset_list, transform, target_transform, co_transform, image_loader)
    test_dataset = ListDataset(test_dataset_list, transform, target_transform, co_transform, image_loader)
    return train_dataset, test_dataset


def image_loader(ref_dir, tar_dir, disp_dir):
    ref_im = imread(ref_dir)
    tar_im = imread(tar_dir)
    disp_im, _ = read_pfm(disp_dir)
    return [ref_im, tar_im], disp_im


def make_dataset(dir, split=80):
    color_dir = os.path.join(dir, 'color')
    disp_dir = os.path.join(dir, 'disp')

    dataset_list = []
    for disp_map in glob.iglob(os.path.join(disp_dir, 'left', '*.pfm')):
        disp_base_name = os.path.basename(disp_map)
        ref_im_base_name = str(disp_base_name[:-4] + ".png")
        ref_im_path = os.path.join(os.path.join(os.path.join(dir, color_dir), 'left'), ref_im_base_name)
        tar_im_path = ref_im_path.replace('left', 'right')
        dataset_list.append([[ref_im_path, tar_im_path], disp_map])

    assert (len(dataset_list) > 0)
    random.shuffle(dataset_list)
    split_index = int(math.floor(len(dataset_list) * split / 100))
    assert (split_index >= 0 and split_index <= len(dataset_list))

    return (dataset_list[:split_index], dataset_list[split_index:]) if split_index < len(dataset_list) else (
        dataset_list, [])


class ListDataset(data.Dataset):
    def __init__(self, dataset_list, transform, target_transform, co_transform, loader=image_loader):
        self.dataset_list = dataset_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, item):
        inputs, target = self.dataset_list[item]
        inputs, target = self.loader(inputs[0], inputs[1], target)

        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def __len__(self):
        return len(self.dataset_list)
