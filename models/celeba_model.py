import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		# 228 x 1 x 1
		self.tconv1 = nn.ConvTranspose2d(228, 448, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(448)
		# 448 x 2 x 2
		self.tconv2 = nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256)
		# 256 x 4 x 4
		self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)
		# 128 x 8 x 8
		self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
		# 64 x 16 x 16
		self.tconv5 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False)
		# 32 x 32 x 32
		self.tconv6 = nn.ConvTranspose2d(32, 16, 4, 2, padding=1, bias=False)
		# 16 x 64 x 64
		self.tconv7 = nn.ConvTranspose2d(16, 3, 4, 2, padding=1, bias=False)
		# 3 x 128 x 128

	def forward(self, x):
		x = F.relu(self.bn1(self.tconv1(x)))
		x = F.relu(self.bn2(self.tconv2(x)))
		x = F.relu(self.tconv3(x))
		x = F.relu(self.tconv4(x))
		x = F.relu(self.tconv5(x))
		x = F.relu(self.tconv6(x))
		img = torch.tanh(self.tconv7(x))
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		# 3 x 128 x 128
		self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)
		# 16 x 64 x 64
		self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
		self.bn2 = nn.BatchNorm2d(32)
		# 32 x 32 x 32
		self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
		self.bn3 = nn.BatchNorm2d(64)
		# 64 x 16 x 16
		self.conv4 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
		self.bn4 = nn.BatchNorm2d(128)
		# 128 x 8 x 8
		self.conv5 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
		self.bn5 = nn.BatchNorm2d(256)
		# 256 x 4 x 4

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
		x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)
		return x


class DHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv = nn.Conv2d(256, 1, 4)

	def forward(self, x):
		output = torch.sigmoid(self.conv(x))

		return output

class QHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
		self.bn1 = nn.BatchNorm2d(128)

		self.conv_disc = nn.Conv2d(128, 100, 1)

		self.conv_mu = nn.Conv2d(128, 1, 1)
		self.conv_var = nn.Conv2d(128, 1, 1)

	def forward(self, x):
		x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

		disc_logits = self.conv_disc(x).squeeze()

		# Not used during training for celeba dataset.
		mu = self.conv_mu(x).squeeze()
		var = torch.exp(self.conv_var(x).squeeze())

		return disc_logits, mu, var

'''
class Generator(nn.Module):
	def __init__(self):
		super().__init__()
		# 228 x 1 x 1
		self.tconv1 = nn.ConvTranspose2d(228, 448, 2, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(448)
		# 448 x 2 x 2
		self.tconv2 = nn.ConvTranspose2d(448, 256, 4, 2, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(256)
		# 256 x 4 x 4
		self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)
		# 128 x 8 x 8
		self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
		# 64 x 16 x 16
		self.tconv5 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)
		# 3 x 32 x 32

	def forward(self, x):
		x = F.relu(self.bn1(self.tconv1(x)))
		x = F.relu(self.bn2(self.tconv2(x)))
		x = F.relu(self.tconv3(x))
		x = F.relu(self.tconv4(x))
		img = torch.tanh(self.tconv5(x))
		return img


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		# 3 x 32 x 32
		self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
		# 64 x 16 x 16
		self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(128)
		# 128 x 8 x 8
		self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(256)
		# 256 x 4 x 4

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
		x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
		return x


class DHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv = nn.Conv2d(256, 1, 4)

	def forward(self, x):
		output = torch.sigmoid(self.conv(x))

		return output

class QHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
		self.bn1 = nn.BatchNorm2d(128)

		self.conv_disc = nn.Conv2d(128, 100, 1)

		self.conv_mu = nn.Conv2d(128, 1, 1)
		self.conv_var = nn.Conv2d(128, 1, 1)

	def forward(self, x):
		x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

		disc_logits = self.conv_disc(x).squeeze()

		# Not used during training for celeba dataset.
		mu = self.conv_mu(x).squeeze()
		var = torch.exp(self.conv_var(x).squeeze())

		return disc_logits, mu, var
'''