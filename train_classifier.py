import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torch.utils.data.sampler import SubsetRandomSampler
from classifiers import LatentSqueezeNet, LatentResnet, resnet18
from dataset import  CodeRow, CodesNpzDataset, RuntimeDataset, RuntimeEvaDataset

from utils import *
from tqdm import tqdm
import json
from torchvision.datasets import CIFAR100

from scheduler import WarmupMultiStepLR
from vqvae import VQVAE

import numpy as np
trans2PIL = transforms.ToPILImage(mode='RGB')
# a special dataset for this
@torch.no_grad()
def test_pure_resnet(epoch, loader, model, device, classes):
	loader = tqdm(loader)

	n_sum = 0
	avg_acc = 0
	acc_sum = 0
	cor_5 =0.0
	cor_1 = 0.0
	model.eval()
	normalized_targets = normalize_classes4cifar(classes)
	for i, (img, label) in enumerate(loader):

		img = img.to(device)
		logits, latent_loss = model(img)
		# label = label.to(device)
		normalized_label = normalized_targets[label]
		label = normalized_label.to(device)
		label = label.to(device)
		# print(logits.size())
		_, winners = (logits).max(1)
		_, pred =logits.topk(5, 1, largest=True, sorted=True)
		# label = torch.zeros_like(label).to(device)
		acc = torch.sum((winners == label).int())
		label = label.view(label.size(0), -1).expand_as(pred)
		correct = pred.eq(label).float()
		cor_5 += correct[:, :5].sum()
		cor_1 += correct[:, :1].sum()
		# print('acc', acc)
		# print('winner', winners)
		# print('label', label)
		acc_sum += acc.detach().item()
		n_sum += img.size(0)
		avg_acc = acc_sum / n_sum
		loader.set_description(
			(
				f"epoch: {epoch + 1};  "
				f" acc:{ avg_acc:.5f} ;"
				f" Top 1 err:{1- cor_1/n_sum:.5f} "
				f" Top 5 err:{1 - cor_5/n_sum:.5f} "


			)
		)

	return avg_acc


def train_pure_resnet(epoch, loader, model, opt, device, classes):
	loader = tqdm(loader)
	criterion = nn.CrossEntropyLoss()
	normalized_targets = normalize_classes4cifar(classes)
	avg_acc = 0
	acc_sum = 0
	mse_sum = 0
	n_sum = 0
	latent_loss_weight = 0.25
	model.train()
	for i, (img,  label) in enumerate(loader):
		model.zero_grad()

		img = img.to(device)


		logits, latent_loss = model(img)
		normalized_label = normalized_targets[label]
		label = normalized_label.to(device)
		label = label.to(device)
		# print(logits.size())
		loss = criterion(logits, label)
		loss = loss + latent_loss_weight * latent_loss
		loss.backward()
		# # if is pretraining the model, update the model
		#
		opt.step()


		_, winners = (logits).max(1)

		# label = torch.zeros_like(label).to(device)
		acc = torch.sum((winners == label).int())


		part_n = img.shape[0]
		part_mse_sum = loss.item() * part_n

		mse_sum += part_mse_sum
		n_sum += part_n
		acc_sum += acc.detach().item()

		avg_acc = acc_sum / n_sum
		lr = opt.param_groups[0]["lr"]
		loader.set_description(
			(
				f"epoch: {epoch + 1};  "
				f" loss:{ mse_sum/n_sum:.5f} "
				f" acc:{ avg_acc:.5f} ;"
				f"lr: {lr:.5f}"
			)
		)



@torch.no_grad()
def test_classifier(epoch, loader, classes, AE, model, device, mode='img'):
	loader = tqdm(loader)
	n_sum = 0
	avg_acc = 0
	acc_sum = 0
	cor_5 = 0.0
	cor_1 = 0.0
	model.eval()
	AE.eval()
	normalized_targets = normalize_classes4cifar(classes)
	for i, (img, label) in enumerate(loader):
		img = img.to(device)
		if mode == 'img':

			logits = model(img)
		elif mode =='latent':
			img = img.to(device)

			_, _, _, id_t, id_b = AE.encode(img)
			quant_b = AE.quantize_b.embed_code(id_b).permute(0, 3, 1, 2)
			quant_t = AE.quantize_t.embed_code(id_t).permute(0, 3, 1, 2)
			upsample_t = AE.upsample_t(quant_t)
			logits = model(upsample_t, quant_b)

		normalized_label = normalized_targets[label]
		normalized_label = normalized_label.to(device)
		_, winners = (logits).max(1)
		_, pred = logits.topk(5, 1, largest=True, sorted=True)
		acc = torch.sum((winners == normalized_label).int())
		normalized_label = normalized_label.view(normalized_label.size(0), -1).expand_as(pred)
		correct = pred.eq(normalized_label).float()
		cor_5 += correct[:, :5].sum()
		cor_1 += correct[:, :1].sum()
		# print('acc', acc)
		acc_sum += acc.detach().item()
		n_sum += img.size(0)
		avg_acc = acc_sum / n_sum
		loader.set_description(
			(
				f"epoch: {epoch + 1}; "
				f" acc:{ avg_acc:.5f} ;"
				f" Top 1 err:{1- cor_1/n_sum:.5f} "
				f" Top 5 err:{1 - cor_5/n_sum:.5f} "
			)
		)
	return avg_acc



# using reconstructed images as inputs
def train_vq_classifier_recon(epoch, loader, classes, AE, model, opt, device):

	loader = tqdm(loader)
	criterion = nn.CrossEntropyLoss()
	avg_acc = 0
	acc_sum = 0
	n_sum = 0
	cor_5 = 0.0
	cor_1 = 0.0
	model.train()
	AE.eval()

	normalized_targets = normalize_classes4cifar(classes)
	transform = transform_ADA_recon_cifar
	for i, (codes_t, codes_b, label) in enumerate(loader):
		label = label.squeeze()
		B = label.size(0)
		model.zero_grad()
		codes_b = codes_b.to(device, dtype=torch.int64)
		codes_t = codes_t.to(device, dtype=torch.int64)
		recon = AE.decode_code(codes_t, codes_b)

		ds = RuntimeDataset(recon.detach().cpu(), transform)
		runtime_loader = DataLoader(ds, B, num_workers=0)
		# BxCxWxH
		transformed_recon = next(iter(runtime_loader)).to(device)

		logits = model(transformed_recon)
		normalized_label = normalized_targets[label].to(device)

		loss = criterion(logits, normalized_label)
		loss.backward()
		opt.step()
		lr = opt.param_groups[0]["lr"]
		_, winners = (logits).max(1)
		_, pred = logits.topk(5, 1, largest=True, sorted=True)
		acc = torch.sum((winners == normalized_label).int())
		normalized_label = normalized_label.view(normalized_label.size(0), -1).expand_as(pred)
		correct = pred.eq(normalized_label).float()
		cor_5 += correct[:, :5].sum()
		cor_1 += correct[:, :1].sum()
		acc_sum += acc.detach().item()
		n_sum += label.size(0)
		avg_acc = acc_sum / n_sum
		loader.set_description(
			(
				f"train epoch: {epoch + 1};  "
				f" loss:{loss.item() :.5f} "
				f" train_acc:{acc_sum / n_sum:.5f} "
				f"lr: {lr:.5f}"
			)
		)

	return avg_acc


# this version uses latent as inputs
def train_vq_classifier_latent(epoch, loader, classes, AE, model, opt, device):
	loader = tqdm(loader)
	criterion = nn.CrossEntropyLoss()
	mse_sum = 0
	n_sum = 0
	acc_sum = 0
	model.train()
	AE.eval()
	normalized_targets = normalize_classes4cifar(classes)
	def flip_code(code):

		return torch.fliplr(code)
	# x is Bx4x4
	for i, (codes_t, codes_b, label) in enumerate(loader):
		opt.zero_grad()
		label = label.squeeze()
		model.zero_grad()
		# perform transform in code level; but it can also be achieved in in latent level
		transformed_cb = torch.ones_like(codes_b)
		transformed_ct = torch.ones_like(codes_t)

		for i, (cb, ct) in enumerate(zip(codes_b, codes_t)):
			p_flip = 0.5
			if torch.rand(1) < p_flip:
				cb = flip_code(cb)
				ct = flip_code(ct)
			# cb = cutout_code(cb)
			# ct = cutout_code(ct)
			transformed_cb[i] = cb
			transformed_ct[i] = ct

		codes_b = transformed_cb.to(device, dtype=torch.int64)
		codes_t = transformed_ct.to(device, dtype=torch.int64)
		quant_b = AE.quantize_b.embed_code(codes_b).permute(0, 3, 1, 2)
		quant_t = AE.quantize_t.embed_code(codes_t).permute(0, 3, 1, 2)
		upsample_t = AE.upsample_t(quant_t)
		logits = model(upsample_t, quant_b)
		normalized_label = normalized_targets[label]
		normalized_label = normalized_label.to(device)
		_, winners = (logits).max(1)
		acc = torch.sum((winners == normalized_label).int())
		acc_sum += acc.detach().item()

		loss = criterion(logits, normalized_label)
		loss.backward()

		opt.step()

		part_n = label.shape[0]
		part_mse_sum = loss.item() * part_n

		mse_sum += part_mse_sum
		n_sum += part_n
		lr = opt.param_groups[0]["lr"]
		loader.set_description(
			(
				f"epoch: {epoch + 1};  "
				f" loss:{mse_sum / n_sum:.5f} "
				f" train_acc:{acc_sum / n_sum:.5f} "
				f"lr: {lr:.5f}"
			)
		)


def test_AE(loader, model, save_path, device):
	sample_size = 6
	for i, (img, label) in enumerate(loader):
		model.eval()
		img = img.to(device)
		sample = img[:sample_size]
		with torch.no_grad():
			out, _ = model(sample)
		utils.save_image(
			torch.cat([sample, out], 0),
			save_path + f"/output/test_AE.png",
			nrow=sample_size,
			normalize=True,
			range=(-1, 1),
		)
		model.train()


def main(args):
	root = args.root
	results_dir = args.results_dir
	save_path = os.path.join(root, results_dir)
	print('root is', root)
	print('save_path is:', save_path)
	os.makedirs(save_path, exist_ok=True)
	json_file_name = os.path.join(save_path, 'args.json')
	with open(json_file_name, 'w') as fp:
		json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)
	checkpoints_path = os.path.join(save_path, 'checkpoints')
	os.makedirs(checkpoints_path, exist_ok=True)
	sample_output_path = os.path.join(save_path, 'output')
	os.makedirs(sample_output_path, exist_ok=True)

	log_file = os.path.join(save_path, 'log.txt')

	config_logging(log_file)
	device = "cuda"
	logging.info('====>  args{} '.format(args))
	num_workers = args.num_workers

	batch_size = args.batch_size
	dataset_path = args.dataset_path
	print('dataset', dataset_path)

	train_ds = CIFAR100(root=dataset_path, train=True, download=False, transform=transform_train_cifar)
	test_ds = CIFAR100(root=dataset_path, train=False, download=False, transform=transform_test_cifar)

	obtain_indices = get_indices
	end_class = args.end_class

	classes = [i for i in range(100)]

	print('data cls', classes)

	testing_idx = obtain_indices(test_ds, classes, is_training=False)
	training_idx = obtain_indices(train_ds, classes, is_training=True)
	codes_path = os.path.join(root, args.codes_path)

	# print('code_path', codes_path)
	#
	# codes_ds = CodesNpzDataset(codes_path)
	# codes_training_idx = obtain_indices(codes_ds, classes, is_training=True)
	#
	# codes_loader = DataLoader(codes_ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, sampler=SubsetRandomSampler(codes_training_idx))
	train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(training_idx), num_workers=num_workers, drop_last=False)
	test_loader = DataLoader(test_ds, batch_size=batch_size, sampler=SubsetRandomSampler(testing_idx), num_workers=num_workers, drop_last=False)



	model = resnet18(3, 100).to(device)


	#
	# mode_pt_path = args.model_pt
	# model_pt = torch.load(mode_pt_path)
	# model.load_state_dict(model_pt)

	opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	# AE = VQVAE(n_embed=args.n_emb, embed_dim=args.dim_emb).to(device)
	# AE_pt_path = os.path.join(root, args.AE_pt)
	# AE_pt = torch.load(AE_pt_path)
	# AE.load_state_dict(AE_pt)

	MILESTONES = [60, 120, 160]
	warmupMultiStepLR = WarmupMultiStepLR(opt, milestones=MILESTONES, gamma=0.2, warmup_iters=args.warm)

	best_acc = 0.0
	scheduler = warmupMultiStepLR

	for i in range(args.epoch):

		train_pure_resnet(i,train_loader, model,opt,device,classes)
		tmp_acc =test_pure_resnet(i,test_loader,model,device,classes)
		if tmp_acc > best_acc:
			best_acc = tmp_acc
			logging.info('====>  Epoch{}: best_acc {} '.format(i, best_acc))
			pt_path = os.path.join(save_path, f"checkpoints/classifier_best.pt")
			torch.save(model.state_dict(), pt_path)
		scheduler.step()


########################################


if __name__ == "__main__":
	parser = argparse.ArgumentParser()


	parser.add_argument("--lr", type=float, default=0.1)
	parser.add_argument("--epoch", type=int, default=200)
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--dataset_path", default='D:\Dataset\cifar100', type=str)
	parser.add_argument("--root", default='', type=str)
	parser.add_argument("--results_dir", default='results', type=str)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--AE_pt", help='path to pretrained VQ-VAE', default='results/checkpoints/VQVAE2_cifar_best.pt', type=str)
	parser.add_argument("--codes_path", default='results/saved_codes/lzma_codes_4.xz', type=str)
	parser.add_argument("--warm", help='warm up epoch', type=int, default=5)
	# parser.add_argument("--dataset", default='core50', type=str)
	parser.add_argument("--end_class", help='number of classes used to train classifier', default=100, type=int)
	parser.add_argument("--n_emb", type=int, help=' the size of codebook, i.e. the number of embeddings', default=512)
	parser.add_argument("--dim_emb", type=int, help='the dimension of the embedding ', default=64)
	parser.add_argument("--input_type", help="'img' for image replay or 'latent' for latent replay", type=str, default='img')
	parser.add_argument("--arch", type=str, help="'resent' for Resnet18 'SQnet' for SqueezeNet", default='resnet')
	parser.add_argument("--is_finetune", help=" True for train classifier using fine-tuning; False for train from scratch ", default=False, type=bool)

	parser.add_argument("--model_pt", help="the path of checkpoints used for fine-tuning; if is_finetune is True or TTA_aug_times not equals to -1 , this must be specified", type=str)
	args = parser.parse_args()

	# print(args)
	main(args)


