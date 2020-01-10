"""ICDAR Dataset Classes

# Author: Fantasy
# Update data: 2020-01-09

"""
import os
import sys
import glob
import torch
import torch.utils.data as data
from PIL import Image
import cv2 as cv
import numpy as np

# note: if you used our download scripts, this should be right
ICDAR_ROOT = "/home/Datasets/TextDet/ICDAR/"


class ICDARDataset(data.Dataset):
    """ICDAR Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): ('2013', 'train'), ('2015', 'train'), ('2017', 'val')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'ICDAR2015')
    """

    def __init__(self, root,
                 image_sets=('2015', 'train'),
                 transform=None,  dataset_name='ICDAR'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.name = dataset_name
        (year, name) = image_sets
        imgpath = os.path.join(self.root, 'ICDAR' + year, name + '_images')
        annopath = os.path.join(self.root, 'ICDAR' + year, name + '_annos')
        nof = len(glob.glob(os.path.join(imgpath, '*.jpg')))
        noa = len(glob.glob(os.path.join(annopath, '*.txt')))
        assert nof == noa
        self.nof = nof

        self._imgpath = os.path.join(imgpath, 'img_%d.jpg')
        self._annopath = os.path.join(annopath, 'gt_img_%d.txt')
        print('Number of images in ICDAR{}_{}: {:d}'.format(year, name, self.nof))
        
    def __getitem__(self, index):       
        index = index + 1
        img = cv.imread(self._imgpath % index, cv.IMREAD_ANYCOLOR)
        height, width = img.shape[:2]
        # img = Image.open(self._imgpath % index).convert('RGB')
        # width, height = img.size
        with open(self._annopath % index, mode='r', encoding='UTF-8-sig') as fp:
            data = fp.read().splitlines()
        target = []
        for idx in data:
            bndbox = []
            spt = idx.split(',')
            x1 = int(spt[0])
            y1 = int(spt[1])
            x2 = int(spt[2])
            y2 = int(spt[3])
            x3 = int(spt[4])
            y3 = int(spt[5])
            x4 = int(spt[6])
            y4 = int(spt[7])
            xmin = min(x1, x2, x3, x4) / width
            ymin = min(y1, y2, y3, y4) / height
            xmax = max(x1, x2, x3, x4) / width
            ymax = max(y1, y2, y3, y4) / height
            text = 0#spt[7]
            target.append([xmin, ymin, xmax, ymax, text])
        if len(target)==0:
            raise Exception('img', index+1)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # BGR to RGB
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
            # HWC to CHW
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def __len__(self):
        return self.nof

    