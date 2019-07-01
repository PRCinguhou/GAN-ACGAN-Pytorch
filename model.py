import torch
import torch.nn as nn


class Generator(nn.Module):

	def __init__(self, latent_dim=100):
		super(Generator, self).__init__()
		self.latent_dim = latent_dim
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, 64 * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
			nn.Tanh(),
			)

	def forward(self, x):
		x = x.view(x.size(0), self.latent_dim, 1, 1)
		img = self.deconv(x)
		img = img.view(x.size(0), 3, 64, 64)
		return img

class Discriminator(nn.Module):

	def __init__(self):
		super(Discriminator, self).__init__()

		self.cnn = nn.Sequential(
			# 3 * 64 * 64
			nn.Conv2d(3, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			# 32 * 64 * 64
			nn.Conv2d(32, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			# 64 * 32 * 32
			nn.Conv2d(64, 128, 3, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			# 128 * 16 * 16
			)
		self.fc = nn.Sequential(
			nn.Linear(128 * 16 * 16, 1024),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
			nn.Linear(1024, 1),
			nn.Sigmoid()
			)

	def forward(self, img):
		
		img = self.cnn(img)
		img = img.view(img.size(0), -1)
		res = self.fc(img)

		return res
