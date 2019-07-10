import torch
import torch.nn as nn


class Generator(nn.Module):

	def __init__(self, latent_dim=100):
		super(Generator, self).__init__()
		self.latent_dim = latent_dim
		self.fc = nn.Sequential(
			nn.Linear(self.latent_dim, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(True),
			nn.Linear(1024, 128 * 16 * 16),
			nn.BatchNorm1d(128*16*16),
			nn.ReLU(True),


			)
		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(128, 64, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 3, 4, 2, 1),
			nn.Tanh(),
			)

	def forward(self, x):
		x = x.view(x.size(0), self.latent_dim)
		x = self.fc(x)
		x = x.view(x.size(0), 128, 16, 16)
		img = self.deconv(x)
		img = img.view(x.size(0), 3, 64, 64)
		return img

class Discriminator(nn.Module):

	def __init__(self):
		super(Discriminator, self).__init__()

		self.cnn = nn.Sequential(
			# 3 * 64 * 64
			nn.Conv2d(3, 32, 4, 2, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2),
			# 32 * 64 * 64
			nn.Conv2d(32, 64, 4, 2, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			)
		self.fc = nn.Sequential(
			nn.Linear(64 * 16 * 16, 512),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(),
			nn.Dropout(0.5),
			nn.Linear(512, 1),
			nn.Sigmoid()
			)

	def forward(self, img):
		
		img = self.cnn(img)
		img = img.view(img.size(0), -1)
		res = self.fc(img)

		return res

class ACGAN_Generator(nn.Module):

	def __init__(self, latent_dim=100):
		super(ACGAN_Generator, self).__init__()

		self.class_num = 2

		self.LinearTransform = nn.Sequential(
			nn.Linear(self.class_num + latent_dim, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(True),
			nn.Linear(1024, 128 * 16 * 16),
			nn.BatchNorm1d(128 * 16 * 16),
			nn.LeakyReLU(True)
			)

		self.generate = nn.Sequential(
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 64, 5, 1, 2),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			
			nn.Upsample(scale_factor=2),
			nn.Conv2d(64, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(True),

			nn.Conv2d(32, 3, 5, 1, 2),
			nn.Tanh(),

			)

	def forward(self, x, label):

		x = torch.cat([x, label], 1)
		x = self.LinearTransform(x)
		x = x.view(-1, 128, 16, 16)
		x = self.generate(x)

		return x

class ACGAN_Discriminator(nn.Module):

	def __init__(self):
		super(ACGAN_Discriminator, self).__init__()

		self.cnn = nn.Sequential(
			# 3 * 64 * 64
			nn.Conv2d(3, 32, 3, 2, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2),
			# 32 * 64 * 64
			nn.Conv2d(32, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			# 64 * 32 * 32
			nn.Conv2d(64, 128, 3, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			# 128 * 16 * 16
			)
		self.fc = nn.Sequential(
			nn.Linear(128 * 8 * 8, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.5),
			nn.Linear(1024, 1),
			nn.Sigmoid()
			)

		self.class_pred = nn.Sequential(
			nn.Linear(128 * 8 * 8, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.5),
			nn.Linear(1024, 2),
			nn.Softmax()
			)

	def forward(self, img):
		img = self.cnn(img)
		img = img.view(img.size(0), -1)

		valid = self.fc(img)
		class_output = self.class_pred(img)

		return valid, class_output
		







