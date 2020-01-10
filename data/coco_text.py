import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np
from .coco_api import *

COCO_ROOT = '/home/Datasets/coco/coco2014'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_TEXT = 'COCO_Text.json'


class COCOTEXTDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """
    def __init__(self, root, image_set='train2014', transform=None, dataset_name='MS COCO_Text'):
        self.root = os.path.join(root, IMAGES, image_set)
        self.ct = COCO_Text(os.path.join(root, ANNOTATIONS, COCO_TEXT))
        self.ids = self.ct.getImgIds(imgIds=self.ct.train, catIds=[('legibility', 'legible'), ('class', 'machine printed')])
        self.transform = transform
        self.name = dataset_name
        self._imgpath = os.path.join(COCO_ROOT, IMAGES, image_set, '%s')
        print('Number of images in COCO_Text: {:d}'.format(len(self.ids)))
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_info = self.ct.loadImgs(self.ids[index])[0]
        path = self._imgpath % img_info['file_name']
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        annIds = self.ct.getAnnIds(imgIds=img_info['id'])
        anns = self.ct.loadAnns(annIds)

        img = cv.imread(path)
        height, width, _ = img.shape
        scale = np.array([width, height, width, height])
        target = []
        for ann in anns:
            if ann['legibility'] == 'legible':
                bbox = ann['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                final_box = list(np.array(bbox) / scale)
                label = 0
                final_box.append(label)
                target += [final_box]
            else:
                continue

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def __len__(self):
        return len(self.ids)
