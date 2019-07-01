from torch.utils.data import Dataset, DataLoader
import os
from os.path import join
from os import listdir
from PIL import Image
from torchvision import transforms
import torch

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])


class GAN_DATASET(Dataset):

	def __init__(self, path):
		self.path = path
		self.images = listdir(join(os.getcwd(), path))

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):	

		img = Image.open(join(os.getcwd(), self.path, self.images[idx]))

		img = transform(img)

		label = torch.FloatTensor([1])

		return img, label



