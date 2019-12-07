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

############# Hyper Parameters #############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
birds_dataset_path = './birds.hdf5'
flowers_dataset_path = './flowers.hdf5'
dataset_name = 'flowers'
save_img_path = './results_right_images/'+dataset_name+'/'
pre_trained_gen = './checkpoints_cls_true/birds_checkpoints/gen_190.pth'
pre_trained_disc = './checkpoints_cls_true/birds_checkpoints/disc_190.pth'
noise_dim = 100
batch_size = 64
num_workers = 8

if not os.path.exists(save_img_path):
	os.makedirs(save_img_path)
############## Model Definition #####################
gen = torch.nn.DataParallel(generator().to(device))
disc = torch.nn.DataParallel(discriminator().to(device))

if pre_trained_gen:
	gen.load_state_dict(torch.load(pre_trained_gen, map_location=device))
else:
	print('Checkpoints file not given!!!')

if pre_trained_disc:
	disc.load_state_dict(torch.load(pre_trained_disc, map_location=device))
else:
	print('Checkpoints file not given!!!')

########### Dataset and Dataloader ####################
if dataset_name == 'birds':
	dataset = Text2ImageDataset(birds_dataset_path, split=3)

if dataset_name == 'flowers':
	dataset = Text2ImageDataset(flowers_dataset_path, split=3)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

############ Testing Code ####################
for sample in data_loader:

	right_images = sample['right_images']
	right_embed = sample['right_embed']
	txt = sample['txt']

	right_images = Variable(right_images.float()).to(device)
	right_embed = Variable(right_embed.float()).to(device)
	
	noise = Variable(torch.randn(right_images.size(0), noise_dim)).to(device)
	noise = noise.view(noise.size(0), noise_dim, 1, 1)
	fake_images = gen(right_embed, noise)

	for image, t in zip(right_images, txt):
		im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
		im.save(save_img_path+t.replace("/", "").replace(".", "").replace("\n","")[:100]+'.jpg')
		print(t)
