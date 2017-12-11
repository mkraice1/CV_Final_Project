#########LOAD LIBRARIES###############               
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
from PIL import Image
######################################

def main():
    data = BodyPoseSet()
    sample_tuple = data[0]
    input_test = sample_tuple['img']
    input_test = input_test.unsqueeze(0)
    net = Body_Net()
    output_test = net(Variable(input_test))

#########DATA LOADER##################
class BodyPoseSet(Dataset):
    """Body pose dataset"""
    def __init__(self, root_dir='./', mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.all_imgs, self.all_labels = self.parse_files()
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label_name= self.all_labels[idx]
        label_path = os.path.join(self.root_dir, label_name)
        img = Image.open(img_path).convert('L')
        label = Image.open(label_path)

        # Transform to tensor
        trans = transforms.Compose([
            transforms.Resize((250,250)),
            transforms.ToTensor(),
        ])
        img = trans(img)
        label = trans(label)

        # Extra transformation
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
            
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


#########NET WORK STRUCTURE###########
class Body_Net(nn.Module):
    def __init__(self):
        super(Body_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU()
        self.conv4_class = nn.Conv2d(256, 44, kernel_size=3, stride=1)
        self.upscore1 = nn.ConvTranspose2d(44, 44, kernel_size=3, stride=1, bias=False)
        self.score_pool2 = nn.Conv2d(128, 44, kernel_size=1, stride=1)
        self.dropout = nn.Dropout2d()  # defualt = 0.5, which is used in paper
        self.upscore2 = nn.ConvTranspose2d(44, 44, kernel_size=4, stride=2, bias=False)
        self.score_pool1 = nn.Conv2d(64, 44, kernel_size=1, stride=1)
        self.upscore3 = nn.ConvTranspose2d(44, 44, kernel_size=19, stride=7, bias=False)

    def forward(self, data):
        h = data
        h = self.relu1(self.conv1(h))
        h = self.pool1(h)
        # record pool 1
        pool1 = h
        h = self.relu2(self.conv2(h))
        h = self.pool2(h)
        # record pool 2
        pool2 = h
        h = self.relu3(self.conv3(h))
        h = self.conv4_class(h)
        h = self.upscore1(h)
        # upsample output
        upscore1 = h
        # crop pool2 and fuse with upscore1
        h =  self.score_pool2(pool2)
        h = h[:, :, 1:17, 1:17]
        score_pool2 = h
        h = upscore1 + score_pool2
        h = self.dropout(h)
        # upsample output
        h = self.upscore2(h)
        upscore2 = h
        # crop pool1 and fuse with upscore2
        h = self.score_pool1(pool1)
        h = h[:, :, 3:37, 3:37]
        score_pool1 = h
        h = upscore2 + score_pool1
        h = self.dropout(h)
        output = self.upscore3(h)

        return output

if __name__ == "__main__":
    main()










