from torch import nn


class BodyNet(nn.Module):

    def __init__(self):
        super(BodyNet, self).__init__()

        # input data, output conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=5)

        # input conv1, output conv1x
        self.relu = nn.ReLU()

        # input conv1x, output pool1
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        # input pool1, output conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # input conv2, output conv2x
        self.relu2 = nn.ReLU()

        # input conv2x, output pool2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input pool2, output conv3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=1)

        # input conv3, output conv3x
        self.relu3 = nn.ReLU(inplace=True)

        # input conv3x, output conv4_class (conv)
        self.conv4_class = nn.Conv2d(256, 44, kernel_size=3, stride=1)

        # input conv4_class, output upscore (convTranspose?)
        self.upscore = nn.ConvTranspose2d(44, 44, kernel_size=3, stride=1)

        # input pool2, output score_pool2 (conv)
        self.score_pool2 = nn.Conv2d(128, 44, kernel_size=1, stride=1)

        # input score_fused, output score4 (convTranspose)
        # dropout 0.5
        self.upsample_fused_16 = nn.ConvTranspose2d(44, 44, kernel_size=4, stride=2)

        # input pool1, output score_pool1 (conv)
        self.score_pool1 = nn.Conv2d(64, 44, kernel_size=1, stride=1)

        # input score_final, output score (convTranspose)
        self.upsample = nn.ConvTranspose2d(44,44, kernel_size=19, stride=7)

        # input score, output prob (SoftMax)
        self.prob = nn.Softmax()

        self.dropout = nn.Dropout()



    def forward(self, data):
        conv1   = self.conv1(data)
        conv1x  = self.relu(conv1)
        pool1   = self.pool1(conv1x)
        conv2   = self.conv2(pool1)
        conv2x  = self.relu(conv2)
        pool2   = self.pool2(conv2x)
        conv3   = self.conv3(pool2)
        conv3x  = self.relu(conv3)
        conv4_class = self.conv4_class(conv3x)
        upscore = self.upscore(conv4_class)

        # crop to 16x16. not sure if this is correct
        # Not sure if we need to crop from the top left (what I did) or the center
        score_pool2c = self.score_pool2(pool2)[:,:,0:16, 0:16]

        # fuse and dropout 0.5
        score_fused = self.dropout(score_pool2c + upscore)
        score4 = self.upsample_fused_16(score_fused)

        # need to crop to 34x34 to fuse with score4.
        # same question as the last crop
        score_pool1c = self.score_pool1(pool1)[:,:,0:34, 0:34]

        # fuse and dropout 0.5
        score_final = self.dropout(score4 + score_pool1c)
        score = self.upsample(score_final)
        prob = self.prob(score)

        return prob


def bodynet(pretrained=False, **kwargs):

    model = BodyNet(**kwargs)
    return model