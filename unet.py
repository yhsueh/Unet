import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Unet(nn.Module):
    def __init__(self, verbose=False):
        self.verbose = verbose
        super(Unet, self).__init__()
        # Contraction Path
        # 1 input image channel (Gray scale), 64 output channels, 3x3 square convolution kernel
        self.down_conv1_1 = nn.Conv2d(1, 64, 3)
        self.down_conv1_2 = nn.Conv2d(64, 64, 3)
        self.down_conv2_1 = nn.Conv2d(64, 128, 3)
        self.down_conv2_2 = nn.Conv2d(128, 128, 3)
        self.down_conv3_1 = nn.Conv2d(128, 256, 3)
        self.down_conv3_2 = nn.Conv2d(256, 256, 3)
        self.down_conv4_1 = nn.Conv2d(256, 512, 3)
        self.down_conv4_2 = nn.Conv2d(512, 512, 3)
        self.down_conv5_1 = nn.Conv2d(512, 1024, 3)
        self.down_conv5_2 = nn.Conv2d(1024, 1024, 3)

        # Expansion Path
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up_conv4_1 = nn.Conv2d(1024, 512, 3)
        self.up_conv4_2 = nn.Conv2d(512, 512, 3)
        self.up_conv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up_conv3_1 = nn.Conv2d(512, 256, 3)
        self.up_conv3_2 = nn.Conv2d(256, 256, 3)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv2_1 = nn.Conv2d(256, 128, 3)
        self.up_conv2_2 = nn.Conv2d(128, 128, 3)
        self.up_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv1_1 = nn.Conv2d(128, 64, 3)
        self.up_conv1_2 = nn.Conv2d(64, 64, 3)
        self.final_conv = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        # level 1
        x = F.relu(self.down_conv1_1(x))
        x = F.relu(self.down_conv1_2(x))
        if self.verbose:
            print('level 1 left shape: ', x.shape) 
        x_crop_1 = x
        x = F.max_pool2d(x, (2, 2))

        # level 2
        x = F.relu(self.down_conv2_1(x))
        x = F.relu(self.down_conv2_2(x))
        if self.verbose:
            print('level 2 left shape: ', x.shape)
        x_crop_2 = x
        x = F.max_pool2d(x, (2, 2))

        # level 3
        x = F.relu(self.down_conv3_1(x))
        x = F.relu(self.down_conv3_2(x))
        if self.verbose:
            print('level 3 left shape: ', x.shape)
        x_crop_3 = x
        x = F.max_pool2d(x, (2, 2))

        # level 4
        x = F.relu(self.down_conv4_1(x))
        x = F.relu(self.down_conv4_2(x))
        if self.verbose:
            print('level 4 left shape: ', x.shape)
        x_crop_4 = x
        x = F.max_pool2d(x, (2, 2))

        # level 5
        x = F.relu(self.down_conv5_1(x))
        x = F.relu(self.down_conv5_2(x))

        # level 4
        x = self.up_conv4(x)
        x = self._crop_paste(x_crop_4, x)
        if self.verbose:
            print('level 4 right shape: ', x.shape)
        x = F.relu(self.up_conv4_1(x))
        x = F.relu(self.up_conv4_2(x))

        # level 3
        x = self.up_conv3(x)
        x = self._crop_paste(x_crop_3, x)
        if self.verbose:
            print('level 3 right shape: ', x.shape)
        x = F.relu(self.up_conv3_1(x))
        x = F.relu(self.up_conv3_2(x))

        # level 2        
        x = self.up_conv2(x)
        x = self._crop_paste(x_crop_2, x)
        if self.verbose:
            print('level 2 right shape: ', x.shape)
        x = F.relu(self.up_conv2_1(x))
        x = F.relu(self.up_conv2_2(x))
        
        #level 1
        x = self.up_conv1(x)
        x = self._crop_paste(x_crop_1, x)
        if self.verbose:
            print('level 1 right shape: ', x.shape)
        x = F.relu(self.up_conv1_1(x))
        x = F.relu(self.up_conv1_2(x))
        x = self.final_conv(x)
        return x

    def _crop_paste(self, left, right):
        left_d = left.shape[2]
        right_d = right.shape[2]
        del_pixel = int((left_d - right_d)/2)

        cropped_left = left[:, :, del_pixel:(left_d - del_pixel), del_pixel:(left_d - del_pixel)]
        combined_tensor = torch.cat((cropped_left, right), 1)

        if self.verbose:
            print('left tensor shape: ', left.shape)
            print('right tensor shape: ', right.shape)
            print('combined_tensor shape:', combined_tensor.shape)
        return combined_tensor