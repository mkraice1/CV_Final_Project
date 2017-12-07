import torch
import torch.nn as nn

class Body_Net(nn.Module):
    def __init__(self):
        super(Body_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, stride=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 44, kernel_size=3, stride=1)
        self.upscore1 = nn.ConvTranspose2d(44, 44, kernel_size=3, stride=1, bias=False)
        self.score_pool2 = nn.Conv2d(128, 44, kernel_size=1)
        self.dropout = nn.Dropout2d()  # defualt = 0.5, which is used in paper
        self.upscore2 = nn.ConvTranspose2d(44, 44, kernel_size=4, stride=2, bias=False)
        self.score_pool1 = nn.Conv2d(64, 44, kernel_size=1)
        self.upscore3 = nn.ConvTranspose2d(44, 44, kernel_size=19, stride=7, bias=False)

    def forward(self, x):
        h = x
        h = self.relu1(self.conv1(h))
        h = self.pool1(h)
        # record pool 1
        pool1 = h
        h = self.relu2(self.conv2(h))
        h = self.pool2(h)
        # record pool 2
        pool2 = h
        h = self.relu3(self.conv3(h))
        h = self.conv4(h)
        h = self.upscore1(h)
        # upsample output
        upscore1 = h
        # crop pool2 and fuse with upscore1
        h =  self.score_pool2(pool2)
        h = h[:, :, 2:17, 2:17]
        score_pool2 = h
        h = upscore1 + score_pool2
        h = self.dropout(h)
        # upsample output
        h = self.upscore2(h)
        upscore2 = h
        # crop pool1 and fuse with upscore2
        h = self.score_pool1(pool1)
        h = h[:, :, 4:37, 4:37]
        score_pool1 = h
        h = upscore2 + score_pool1
        h = self.dropout(h)
        output = self.upscore3(h)

        return output

############################################
# In trainer use softmax loss#
############################################











