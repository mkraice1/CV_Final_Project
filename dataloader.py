#!/usr/bin/env python

import torch
import os
import cv2
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
from torch.autograd import Variable
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Define dataset class
# Usage: dataset = BodyPoseSet()
#        sample = dataset[1]
#        print sample
class BodyPoseSet(Dataset):
    """Body pose dataset."""

    def __init__(self, root_dir='./', mode='train'):
        self.root_dir = root_dir
        self.mode = mode
        self.all_imgs, self.all_labels = self.parse_files()

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        # Get images
        img_name = self.all_imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label_name = self.all_labels[idx]       
        label_path = os.path.join(self.root_dir, label_name)
        img = cv2.imread(img_path,0)
        label = cv2.imread(label_path,0)
        sample = {'img':img, 'label':label}
        
        return sample
    
    def parse_files(self):
        all_imgs = []
        all_labels = []
        for a in ['easy-pose']:
            for b in [i+1 for i in range(1)]:
                for c in ['Cam1','Cam2','Cam3']:
                    for d in ["{0:04}".format(i+1) for i in range(1001)]:
                        img_name = "%s/%s/%d/images/depthRender/%s/mayaProject.00%s.png" %(a,self.mode,b,c,d) 
                        all_imgs.append(img_name)
                        label_name = "%s/%s/%d/images/groundtruth/%s/mayaProject.00%s.png" %(a,self.mode,b,c,d)
                        all_labels.append(label_name)
        return all_imgs, all_labels