import torch 
import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import os 
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from argparse import ArgumentParser
from model import ACGAN_Generator, ACGAN_Discriminator
from load_data import GAN_DATASET

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else 'cpu')


parser = ArgumentParser()
parser.add_argument("-EPOCH", "--EPOCH", dest="epoch", type=int, default=100)
parser.add_argument("-batch", "--batch", dest="batch_size", type=int, default=30)
parser.add_argument("-latent_dim", dest='latent_dim', type=int, default=100)
parser.add_argument("-model", dest='model', type=str, default='gan')
parser.add_argument("-dataset", dest='dataset', type=str, default='./face')
parser.add_argument("-lr", dest='lr', type=float, default=1e-4)



args = parser.parse_args()

fixed_noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (32, args.latent_dim)))).to(device)
if not os.path.isfile('fixed.npy'):
	np.save('fixed.npy', fixed_noise.cpu())


if __name__ == '__main__':



	print('''
Current Parameters:\n
Epoch : [%d],\n
Batch size : [%d],\n
Latent Dimension  : [%d],\n
Train model : %s,\n
''' % (args.epoch, args.batch_size, args.latent_dim, args.model))

	loss = nn.BCELoss()

	generator = ACGAN_Generator(args.latent_dim).to(device)
	discriminator = ACGAN_Discriminator().to(device)

	dataloader = GAN_DATASET('face', mode='acgan', feature='Smiling')
	dataloader = DataLoader(dataloader, batch_size=args.batch_size, shuffle=True)

	g_optim = optim.Adam(generator.parameters(), lr=args.lr)
	d_optim = optim.Adam(discriminator.parameters(), lr=args.lr)

		

	for ep in range(args.epoch):

		g_avg_loss = 0
		d_avg_loss = 0

		for step, batch in enumerate(dataloader):

			x, y = batch
			x = x.to(device)
			y = y.to(device)

			valid = torch.FloatTensor(x.size(0), 1).fill_(1.0).to(device)
			fake = torch.FloatTensor(x.size(0), 1).fill_(0.0).to(device)


			#### --------------- ####
			#### train Generator ####
			#### --------------- ####
			g_optim.zero_grad()

			z = torch.FloatTensor(np.random.normal(0, 1, (x.size(0), args.latent_dim))).to(device)

			gen_image = generator(z, y)
			validity, class_pred = discriminator(gen_image)

			valid_loss = loss(validity, valid)
			class_loss = loss(y, class_pred)

			g_loss = valid_loss + class_loss
			g_loss.backward(retain_grapg=True)

			g_optim.step()

			#### ------------------- ####
			#### train Discriminator ####
			#### ------------------- ####

			d_optim.zero_grad()

			
			























