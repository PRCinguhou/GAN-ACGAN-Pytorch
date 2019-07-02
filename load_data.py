from torch.utils.data import Dataset, DataLoader
import os
from os.path import join
from os import listdir
from PIL import Image
from torchvision import transforms
import torch
import pandas as pd

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])


class GAN_DATASET(Dataset):

	def __init__(self, path, mode='gan', feature=None):
		self.mode = mode
		self.path = path
		self.images = listdir(join(os.getcwd(), path, 'train'))
		if mode == 'acgan':
			self.csv_data = pd.read_csv(join(os.getcwd(), path, 'train.csv'))
			self.feature_index = list(self.csv_data.columns).index(feature)
			self.csv_data = self.csv_data.values

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):	


		img = Image.open(join(os.getcwd(), self.path, 'train', self.images[idx]))

		img = transform(img)
		if self.mode == 'gan':
			label = torch.FloatTensor([1])
		elif self.mode == 'acgan':
			label = torch.FloatTensor([0, 0])
			label[self.csv_data[idx][self.feature_index]] = 1

		return img, label



if __name__ == '__main__':
	dataloader = GAN_DATASET('face')
	dataloader = DataLoader(dataloader, batch_size=5)
	for index, i in enunmerate(dataloader):
		x, y = i
		if index == 5:
			break