import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Contraction Path
        # 1 input image channel (Gray scale), 64 output channels, 3x3 square convolution kernel
        self.down_conv1_1 = nn.Conv2d(1, 64, 3)
        self.down_conv1_2 = nn.Conv2d(64, 64, 3)
        self.down_conv2_1 = nn.Conv2d(64, 128, 3)
        self.down_conv2_2 = nn.Conv2d(128, 128, 3)
        self.valley_conv1 = nn.Conv2d(128, 256, 3)
        self.valley_conv2 = nn.Conv2d(256, 256, 3)

        # Expansion Path
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2_1 = nn.Conv2d(128, 128, 3) #Encoder
        self.up_conv2_2 = nn.Conv2d(128, 128, 3)
        self.up_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv1_1 = nn.Conv2d(64, 64, 3) #Encoder
        self.up_conv1_2 = nn.Conv2d(64, 64, 3)
        self.final_conv = nn.Conv2d(64, 2, 1)

    def forward(self, x):

        x = F.relu(self.down_conv1_1(x))
        x = F.relu(self.down_conv1_2(x))
        x = F.max_pool2d(x, (2, 2))

        x = F.relu(self.down_conv2_1(x))
        x = F.relu(self.down_conv2_2(x))
        x = F.max_pool2d(x, (2, 2))
       
        x = F.relu(self.valley_conv1(x))
        x = F.relu(self.valley_conv2(x))

        x = self.up_conv3(x)
        x = F.relu(self.up_conv2_1(x))
        x = F.relu(self.up_conv2_2(x))

        x = self.up_conv2(x)
        x = F.relu(self.up_conv1_1(x))
        x = F.relu(self.up_conv1_2(x))
        x = self.final_conv(x)
        print(x.shape)
        return x