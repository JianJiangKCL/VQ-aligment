import numpy as np
import joblib

from vqvae import  VQVAE
import os
from dataset import CodesNpzDataset, RuntimeDataset
import torch
from torch.utils.data import DataLoader
from utils import  *
from torchvision import datasets, utils
device ='cuda'

# for i, (codes_t, codes_b, label) in enumerate(codes_loader):
# 	label = label.squeeze()
# 	B = label.size(0)
# 	codes_b = codes_b.to(device, dtype=torch.int64)
# 	codes_t = codes_t.to(device, dtype=torch.int64)
# 	recon = AE.decode_code(codes_t, codes_b)
#
# 	ds = RuntimeDataset(recon.detach().cpu(), transform_test_cifar, type='latent')
# 	runtime_loader = DataLoader(ds, B, num_workers=0)
# 	# BxCxWxH
# 	transformed_recon = next(iter(runtime_loader)).to(device)
# 	sampleWOdenorm = transformed_recon[:sample_size]
#
# 	ds = RuntimeDataset(recon.detach().cpu(), transform_test_cifar, type='RGB')
# 	runtime_loader = DataLoader(ds, B, num_workers=0)
# 	transformed_recon = next(iter(runtime_loader)).to(device)
# 	samplewithdenorm = transformed_recon[:sample_size]
#
# 	utils.save_image(
# 		torch.cat([sampleWOdenorm, samplewithdenorm], 0),
# 		save_path + f"/output/test_AE.png",
# 		nrow=sample_size,
# 		normalize=True,
# 		range=(-1, 1),
# 	)
# 	break
from classifiers import  resnet18
model = resnet18(3,100).to(device)
def test_AE(loader, model, device):
	sample_size = 12
	for i, (img, label) in enumerate(loader):
		model.eval()
		img = img.to(device)
		sample = img[:sample_size]

		out = model(sample)

from torchvision.datasets import CIFAR100
dataset_path = "D:\Dataset\cifar100"
train_ds = CIFAR100(root=dataset_path, train=True, download=False, transform=transform_test_cifar)
loader = DataLoader(train_ds, batch_size=128,
							 num_workers=0, drop_last=False)
test_AE(loader, model, device)