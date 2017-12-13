###########LOAD LIBRARIES###############
#########PUBLIC LIBRARIES#############
import torch                                         
import os                                            
import cv2
import argparse                                           
import torch.nn as nn                                
import random                                        
import torch.nn.functional as F                      
import torchvision.datasets as dset                  
import torchvision.transforms as transforms           
import torchvision.utils as utils                    
import numpy as np
import pickle                                   
from torch.autograd import Variable                  
from torch import optim                              
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#########OUR FUNCTIONS###############
from Process_Label import color_to_classes

########## MAIN DRIVER PROGRAM ############
def main():
    parser = argparse.ArgumentParser(description='Body pose recognition.')
    parser.add_argument('--load', type = str, help = 'Using trained parameter to test on both train and test sets.')
    parser.add_argument('--save', type = str, help = 'Train the model using splitting file provided.')
    args = parser.parse_args()
    weights_dir = args.save

    # Setting up configuration
    configs = {"batch_train": 16, \
                "batch_test": 4, \
                "epochs": 2, \
                "num_workers": 4, \
                "learning_rate": 1e-6, \
                "data_augment": True}

   # Training process setup
    img_trans = transforms.Compose([transforms.Resize((250,250)),transforms.ToTensor()])
    label_trans = transforms.Compose([transforms.ToTensor()])
    body_train = BodyPoseSet(img_transform=img_trans, label_transform = label_trans)
    train_loader = DataLoader(body_train, batch_size=configs['batch_train'], shuffle=True, num_workers=configs['num_workers'])

    # Training the net
    net = Body_Net().cuda()
    optimizer = optim.Adam(net.parameters(), lr = configs['learning_rate'])
    total_epoch = configs['epochs']
    counter = []
    loss_history = []
    iteration = 0 
    loss_fn = Cross_Entropy_Loss()

    for epoch in range(total_epoch):
        for batch_idx, batch_sample in enumerate(train_loader):
            img = batch_sample['img']
            label = batch_sample['label']
            img1, y = Variable(img).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            y_pred = net(img1)

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            if batch_idx % (len(body_train)/configs['batch_train']/500) == 0:
                print ("Epoch %d, Batch %d Loss %f" % (epoch, batch_idx, loss.data[0]))
                iteration += 20
                counter.append(iteration)
                loss_history.append(loss.data[0])
        
    # Save the trained network
    torch.save(net.state_dict(), weights_dir)
    total_hist = [counter, loss_history]
    with open("training_history.txt", "wb") as fp:
        pickle.dump(total_hist, fp)
    

#########DATASET CLASS##################
class BodyPoseSet(Dataset):
    """Body pose dataset"""
    def __init__(self, root_dir='./', mode='train', img_transform=None, label_transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.all_imgs, self.all_labels = self.parse_files()
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_name = self.all_imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        label_name= self.all_labels[idx]
        label_path = os.path.join(self.root_dir, label_name)
        img = Image.open(img_path).convert('L')
        label = Image.open(label_path)
        label = color_to_classes(label)

        
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.label_transform is not None:
            label = self.label_transform(label)
            
        sample = {'img':img, 'label':label}

        return sample

    def parse_files(self):
        all_imgs = []
        all_labels = []
        for a in ['easy-pose']:
            for b in [i+1 for i in range(35)]:
                #if b == 36 or b == 106 or b == 178 or b == 72:
                    #continue
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
        self.prob = nn.Softmax2d()

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
        # compute cross entropy 
        # output = self.prob(output)
        # output = -torch.log(output)
        return output

class Cross_Entropy_Loss(nn.Module):

    def __init__(self, weight=None, size_average=True):
        super(Cross_Entropy_Loss, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, y_pred, y):
        """
        y_pred: 16(b) by 44(c) by 250(h) by 250(w)
        y: 16(b) by 250(h) by 250(w)
        """
        n, c, h, w = y_pred.size()
        log_p = F.log_softmax(y_pred)
        log_p = log_p.transpose(1,2).transpose(2,3).contiguous().view(-1,c)
        log_p = log_p[y.view(n,h,w,1).repeat(1,1,1,c)>=0]
        log_p = log_p.view(-1,c)
        mask = y >= 0
        y = y[mask]
        loss = F.nll_loss(log_p, y.type(torch.cuda.LongTensor), weight=self.weight, size_average=False)
        if self.size_average:
            loss /= mask.data.sum()
        
        return loss

if __name__ == "__main__":
    main()
###END OF PROGRAM###









