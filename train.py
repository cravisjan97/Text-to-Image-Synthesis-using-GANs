import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import Text2ImageDataset
from model import generator, discriminator
#from utils import Utils, Logger
from PIL import Image
import os

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def smooth_label(tensor, offset):
	return tensor + offset

############# Hyper Parameters #############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
birds_dataset_path = '/content/drive/My Drive/EECS_595_Project/birds.hdf5'
flowers_dataset_path = '/content/drive/My Drive/EECS_595_Project/flowers.hdf5'
checkpoints_path = '/content/drive/My Drive/EECS_595_Project/checkpoints/'
dataset_name = 'birds'
pre_trained_gen = ''
pre_trained_disc = ''
noise_dim = 100
batch_size = 64
num_workers = 8
lr = 0.0002
epochs = 200
beta1 = 0.5
l1_coef = 50 
l2_coef = 100
cls = True

############## Model Definition #####################
gen = torch.nn.DataParallel(generator().to(device))
disc = torch.nn.DataParallel(discriminator().to(device))

if pre_trained_gen:
	gen.load_state_dict(torch.load(pre_trained_gen))
else:
	weights_init(gen)

if pre_trained_disc:
	disc.load_state_dict(torch.load(pre_trained_disc))
else:
	weights_init(disc)

########### Dataset and Dataloader ####################
if dataset_name == 'birds':
	dataset = Text2ImageDataset(birds_dataset_path, split=0)

if dataset_name == 'flowers':
	dataset = Text2ImageDataset(flowers_dataset_path, split=0)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

########## Losses and Optimizers ######################
optimD = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

criterion = nn.BCELoss()
l2_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

############ Training Code ####################
for epoch in range(num_epochs):
	if (epoch) % 10 == 0:
		print('Saving Checkpoints')
		torch.save(generator.state_dict(), checkpoints_path+dataset_name+'/gen_'+str(epoch)+'.pth')
		torch.save(discriminator.state_dict(), checkpoints_path+dataset_name+'/disc_'+str(epoch)+'.pth')

	start = time.time()
	print('Epoch:{}/{}'.format(epoch+1, num_epochs))

	for i,sample in enumerate(data_loader):
		if i%100==0:
			print ('Iteration:{}/{}'.format(i, len(dataset)//batch_size))
		iteration += 1
		right_images = sample['right_images']
		right_embed = sample['right_embed']
		wrong_images = sample['wrong_images']

		right_images = Variable(right_images.float()).to(device)
		right_embed = Variable(right_embed.float()).to(device)
		wrong_images = Variable(wrong_images.float()).to(device)

		real_labels = torch.ones(right_images.size(0))
		fake_labels = torch.zeros(right_images.size(0))

		smoothed_real_labels = torch.FloatTensor(smooth_label(real_labels.numpy(), -0.1))

		real_labels = Variable(real_labels).to(device)
		smoothed_real_labels = Variable(smoothed_real_labels).to(device)
		fake_labels = Variable(fake_labels).to(device)

		# Train the discriminator
		discriminator.zero_grad()
		outputs, activation_real = discriminator(right_images, right_embed)
		real_loss = criterion(outputs, smoothed_real_labels)
		real_score = outputs

		if cls:
		    outputs, _ = discriminator(wrong_images, right_embed)
		    wrong_loss = criterion(outputs, fake_labels)
		    wrong_score = outputs

		noise = Variable(torch.randn(right_images.size(0), noise_dim)).to(device)
		noise = noise.view(noise.size(0), noise_dim, 1, 1)
		fake_images = generator(right_embed, noise)
		outputs, _ = discriminator(fake_images, right_embed)
		fake_loss = criterion(outputs, fake_labels)
		fake_score = outputs

		d_loss = real_loss + fake_loss

		if cls:
		    d_loss = d_loss + wrong_loss

		d_loss.backward()
		optimD.step()

		# Train the generator
		generator.zero_grad()
		noise = Variable(torch.randn(right_images.size(0), noise_dim)).to(device)
		noise = noise.view(noise.size(0), noise_dim, 1, 1)
		fake_images = generator(right_embed, noise)
		outputs, activation_fake = discriminator(fake_images, right_embed)
		_, activation_real = discriminator(right_images, right_embed)

		activation_fake = torch.mean(activation_fake, 0)
		activation_real = torch.mean(activation_real, 0)

		g_loss = criterion(outputs, real_labels) \
			 + l2_coef * l2_loss(activation_fake, activation_real.detach()) \
			 + l1_coef * l1_loss(fake_images, right_images)

		g_loss.backward()
		optimG.step()

	end = time.time()
	dur = (end-start)
	print('G_loss:{} D_loss:{} Time:{}m{}s'.format(g_loss, d_loss, dur//60, dur%60))
