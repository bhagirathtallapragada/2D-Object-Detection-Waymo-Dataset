import torch 
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
# from data_aug.bbox_util import draw_rect
# from data_aug.data_aug import *
import time
import random
from torch.utils.data import DataLoader
# from bbox import corner_to_center, center_to_corner, bbox_iou
import cv2 
import os

random.seed(0)

class CustomDataset(Dataset):
    def __init__(self, root, num_classes, ann_file, det_transforms=None):
        """Note:  When using VoTT and exported to YOLO, 
        ann_file is a list to paths of images"""
        self.root = root
        
        self.examples = None
        with open(ann_file, 'r') as f:
            self.examples = f.readlines()
        self.det_transforms = det_transforms

        # The following, user needs to modify (TODO - create from args)
        self.inp_dim = 416
        self.strides = [32,16]
        self.anchor_nums = [3,3]
        self.num_classes = num_classes
        self.anchors = [[10,14],  [23,27],  [37,58],  [81,82],  [135,169],  [344,319]]
        
        self.anchors = np.array(self.anchors)[::-1]
        
        #Get the number of bounding boxes predicted PER each scale 
        self.num_pred_boxes = self.get_num_pred_boxes()
        
        self.box_strides = self.get_box_strides()
        self.debug_id = None

    def __len__(self):
        return len(self.examples)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels
        """

        images = list()
        boxes = list()

        for b in batch:
            if len(b[1]) > 0:
                images.append(b[0])
                boxes.append(b[1])
        
        if len(boxes) > 0:
            boxes = torch.stack(boxes, dim=0)
            images = torch.stack(images, dim=0)
        
        return images, boxes