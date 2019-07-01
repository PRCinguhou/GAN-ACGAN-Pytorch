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
from model import Generator, Discriminator
from load_data import GAN_DATASET

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else 'cpu')
# np.save('fixed.npy', fixed_noise.cpu())

parser = ArgumentParser()
parser.add_argument("-EPOCH", "--EPOCH", dest="epoch", type=int, default=100)
parser.add_argument("-batch", "--batch", dest="batch_size", type=int, default=30)
parser.add_argument("-latent_dim", dest='latent_dim', type=int, default=100)
parser.add_argument("-model", dest='model', type=str, default='gan')


args = parser.parse_args()

fixed_noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (32, args.latent_dim)))).to(device)


if __name__ == '__main__':

	print('''
Current Parameters:\n
Epoch : [%d],\n
Batch size : [%d],\n
Latent Dimension  : [%d],\n
Train model : %s,\n
''' % (args.epoch, args.batch_size, args.latent_dim, args.model))


	generator = Generator(args.latent_dim).to(device)
	discriminator = Discriminator().to(device)

	loss = nn.BCELoss()
	
	dis_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
	gen_optim = optim.Adam(generator.parameters(), lr=1e-4)

	dataloader = GAN_DATASET('./face/train')
	dataloader = DataLoader(dataloader, batch_size=args.batch_size, shuffle=True)


	g_avg_loss = 0
	d_avg_loss = 0
	

	for ep in range(args.epoch):
		print('EPOCH : [%d]/[%d]' % (ep, args.epoch))
		img_list = []

		for step, batch in enumerate(dataloader):

			x, y = batch
			x = x.to(device)
			y = y.to(device)
			
			#### train Generator ####
			gen_optim.zero_grad()

			z = Variable(torch.FloatTensor(np.random.normal(0, 1, (x.size(0), args.latent_dim)))).to(device)
			gen_img = generator(z)
			fake = torch.Tensor(x.size(0), 1).fill_(0.0)

			g_loss += loss(discriminator(gen_img), fake)
			g_avg_loss += g_loss.item()

			g_loss.backward()
			gen_optim.step()

			#### train Discriminator ####
			dis_optim.zero_grad()

			real_loss = loss(discriminator(x), y)
			fake_loss = loss(discriminator(gen_img), fake)	

			loss = real_loss + fake_loss
			d_avg_loss += loss.item()


			loss.backward()
			dis_optim.step()


			if step % 50 == 0:
				print("[%d]/[%d] Finished, Generator AVG loss : [%d], Discriminator AVG loss : [%d]" % \
					(step, len(dataloader), g_avg_loss/ args.batch_size/(step+1)), d_avg_loss/args.batch_size/(step+1))


		with torch.no_grad():
			fake = generator(fixed_noise).detach().cpu()

		img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
		fig = plt.figure(figsize=(8,8))
		plt.axis("off")
		ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
		plt.savefig('./gan_result/ep_'+str(ep)+'.jpg')

		torch.save(generator.state_dict(), 'generator.pth')
		torch.save(discriminator.state_dict(), 'discriminator.pth')
		











