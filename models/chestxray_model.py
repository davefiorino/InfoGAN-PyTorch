import torch
import torch.nn as nn
import torch.nn.functional as F
import config

"""
Architecture by Davide Fiorino.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 112 x 1 x 1
        self.tconv1 = nn.ConvTranspose2d(112, 1024, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        # 1024 x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        # 512 x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        # 256 x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        # 128 x 32 x 32
        self.tconv5 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        # 64 x 64 x 64
        self.tconv6 = nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False)
        # 32 x 128 x 128
        self.tconv7 = nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False)
        # 1 x 256 x 256

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.relu(self.tconv5(x))
        x = F.relu(self.tconv6(x))

        img = torch.sigmoid(self.tconv7(x))

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 256 x 256
        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)
        self.drop1 = nn.Dropout(0.25)
        # 32 x 128 x 128
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.25)
        # 64 x 64 x 64
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.25)
        # 128 x 32 x 32
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(0.25)
        # 256 x 16 x 16
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        # 512 x 8 x 8
        self.conv6 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(1024)
        # 1024 x 4 x 4

    def forward(self, x):
        if config.currentEpoch < (config.params['num_epochs']/2) and config.params['use_dropout'] == True: # Use dropout only in the first half of training epochs
            x = self.drop1(F.leaky_relu(self.conv1(x), 0.1, inplace=True))
            x = self.drop2(F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True))
            x = self.drop3(F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True))
            x = self.drop4(F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True))
        else:
            x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
            x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
            x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
            x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)

        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.1, inplace=True)

        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1024, 1, 4)
        # 1 x 1 x 1

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))
        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var
        # [10], [2], [2] 
