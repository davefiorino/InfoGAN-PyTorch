import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture by Davide Fiorino.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # 146 x 1 x 1
        self.tconv1 = nn.ConvTranspose2d(146, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        # 1024 x 1 x 1
        self.tconv2 = nn.ConvTranspose2d(1024, 128, kernel_size=7, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # 128 x 7 x 7
        self.tconv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        # 128 x 14 x 14
        self.tconv4 = nn.ConvTranspose2d(128, 128, 4, 2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        # 128 x 28 x 28
        self.tconv5 = nn.ConvTranspose2d(128, 128, 4, 2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        # 128 x 56 x 56
        self.tconv6 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        # 64 x 112 x 112
        self.tconv7 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)
        # 1 x 224 x 224

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.relu(self.bn5(self.tconv5(x)))
        x = F.relu(self.bn6(self.tconv6(x)))

        img = torch.sigmoid(self.tconv7(x))

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 x 224 x 224
        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        # 64 x 112 x 112
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        # 128 x 56 x 56
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        # 128 x 28 x 28
        self.conv4 = nn.Conv2d(128, 128, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        # 128 x 14 x 14
        self.conv5 = nn.Conv2d(128, 128, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        # 128 x 7 x 7
        self.conv6 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn6 = nn.BatchNorm2d(1024)
        # 1024 x 1 x 1

    def forward(self, x):
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

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 20, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var

