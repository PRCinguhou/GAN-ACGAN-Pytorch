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
parser.add_argument("-model", dest='model', type=str, default='acgan')
parser.add_argument("-dataset", dest='dataset', type=str, default='./face')
parser.add_argument("-lr", dest='lr', type=float, default=1e-4)



args = parser.parse_args()

fixed_noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (20, args.latent_dim)))).to(device)
zero_label = np.vstack([[1, 0] for i in range(10)])
one_lavbel = np.vstack([[0, 1] for i in range(10)])
fixed_label = torch.FloatTensor(np.vstack([zero_label, one_lavbel])).to(device)
if not os.path.isfile('ac_fixed.npy'):
	np.save('ac_fixed.npy', fixed_noise.cpu())


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
		
		print('EPOCH : [%d]/[%d]' % (ep, args.epoch))
		img_list = []
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
			class_loss = loss(class_pred, y)
			g_loss = valid_loss + class_loss
			
			g_avg_loss += g_loss.item()

			g_loss.backward(retain_graph=True)
			g_optim.step()



			#### ------------------- ####
			#### train Discriminator ####
			#### ------------------- ####
		
			d_optim.zero_grad()

			real_valid, real_class = discriminator(x)
			fake_valid, fake_class = discriminator(gen_image)

			real_loss = loss(real_valid, valid) + loss(real_class, y)
			fake_loss = loss(fake_valid, fake) + loss(fake_class, y)

			all_loss = real_loss + fake_loss
			
			d_avg_loss += all_loss.item()

			all_loss.backward(retain_graph=True)
			d_optim.step()


			if step % 50 == 0:
				print("[%d]/[%d] Finished, Generator AVG loss : [%.4f], Discriminator AVG loss : [%.4f]" % \
					(step, len(dataloader), g_avg_loss/(step+1), d_avg_loss/(step+1)))


		if ep % 5 == 0:
			with torch.no_grad():
				fake = generator(fixed_noise, fixed_label).detach().cpu()

			img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
			fig = plt.figure(figsize=(8,8))
			plt.axis("off")
			ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
			plt.savefig('./acgan_result/ep_'+str(ep)+'.jpg')

			torch.save(generator.state_dict(), 'ac_generator.pth')
			torch.save(discriminator.state_dict(), 'ac_discriminator.pth')
			

		




























