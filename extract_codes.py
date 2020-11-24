import argparse
from torch.utils.data import DataLoader

from torchvision import datasets, utils

from utils import *
from tqdm import tqdm
from dataset import ImageNet_downsampled
from torchvision.datasets import CIFAR100
import numpy as np
from vqvae import VQVAE
import joblib

def extract_codes( loader, model, device,  num_embed, save_path, img_size=32, mode='deflate'):
	pbar = tqdm(loader)
	length = 0
	codes_b_array = []
	codes_t_array = []
	label_array = []
	if img_size == 32:
		for img,  label in pbar:
			length += img.size(0)
			img = img.to(device)
			# the indices of embedding units in the map. from 0 to N_e -1
			inputs = img
			_, _, _, id_t, id_b = model.encode(inputs)
			id_b = id_b.detach().cpu()
			id_t = id_t.detach().cpu()
			for code_t, code_b, cls in zip(id_t, id_b, label):
				codes_t_array.append(code_t.numpy())
				codes_b_array.append(code_b.numpy())
				label_array.append(cls.numpy())
	elif img_size == 64:
		for img, label in pbar:
			length += img.size(0)
			B, C, H, _ = img.size()
			tmp = img.to(device).view(B * C, H, H)
			sample_1 = tmp[:, 0:32, 0:32]
			sample_2 = tmp[:, 32:64, 0:32]
			sample_3 = tmp[:, 0:32, 32:64]
			sample_4 = tmp[:, 32:64, 32:64]
			# 4 x B*C x hxw => 4 * Bx C x hxw
			inputs = torch.stack([sample_1, sample_2, sample_3, sample_4], dim=0).view(4 * B, C, 32, 32)
			# 4*B x dh x dw
			_, _, _, id_t, id_b = model.encode(inputs)
			top_h = id_t.size(1)
			bot_h = id_b.size(1)
			# tmp_ct = torch.zeros([B, top_h,top_h])
			# 4xbx dh x dw
			id_t = id_t.view(4, B, top_h, top_h)
			id_b = id_b.view(4, B, bot_h, bot_h)
			# pdb.set_trace()
			id_b = id_b.detach().cpu()
			id_t = id_t.detach().cpu()
			for cnt in range(B):
				# 4x dh x dw
				code_t = id_t[:, cnt, :, :]
				code_b = id_b[:, cnt, :, :]
				codes_t_array.append(code_t.numpy())
				codes_b_array.append(code_b.numpy())
				cls = label[cnt]
				label_array.append(cls.numpy())

	if num_embed > 256:
		data_type = np.int16
	else:
		data_type = np.uint8
	codes_b_array = np.array(codes_b_array, dtype=data_type)
	codes_t_array = np.array(codes_t_array, dtype=data_type)
	label_array = np.array(label_array, dtype=np.uint8)
	# print(mode)
	if mode == 'deflate':
		print('deflate')
		path = os.path.join(save_path, 'deflate_codes')
		np.savez_compressed(path, code_t=codes_t_array, code_b=codes_b_array, label=label_array, length=length)
	elif mode == 'original':
		print('original')
		path = os.path.join(save_path, 'codes')
		np.savez(path, code_t=codes_t_array, code_b=codes_b_array, label=label_array, length=length)
	elif mode == 'lzma':
		print('lzma')
		to_persist = {'code_b': codes_b_array, 'code_t': codes_t_array, 'label': label_array}
		degree = 4
		path = os.path.join(save_path, 'lzma_codes_')
		joblib.dump(to_persist, path+str(degree)+'.xz', compress=('lzma', degree))


def test_AE(loader, model, save_path, device):
	sample_size = 12
	for i, (img, label) in enumerate(loader):
		model.eval()
		img = img.to(device)
		sample = img[:sample_size]
		with torch.no_grad():
			out, _= model(sample)
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

	sample_output_path = os.path.join(save_path, 'output')
	os.makedirs(sample_output_path, exist_ok=True)

	device = "cuda"

	dataset_path = args.dataset_path

	if args.dataset == 'cifar100':
		extract_train_ds = CIFAR100(root=dataset_path, train=True, download=False, transform=transform_test_cifar)

	else:
		if args.img_size == 32:
			extract_train_ds = ImageNet_downsampled('D:\Dataset\Imagenet32_train_npz', idx=1,transform=transform_Img32)
		elif args.img_size == 64:
			extract_train_ds = ImageNet_downsampled('D:\Dataset\Imagenet64_train_npz', idx=1, transform=transform_Img64)

	# As the VQ-VAE is frozen, we can extract codes for all classes at one time for convenient

	extract_loader = DataLoader(extract_train_ds, batch_size=128, shuffle=False, num_workers=0, drop_last=False)

	model = VQVAE().to(device)
	model_pt = torch.load(args.model_pt)
	model.load_state_dict(model_pt)
	test_AE(extract_loader, model, save_path, device)
	img_size = args.img_size
	# mode is the method to save codes; 'original' means no compression.
	extract_codes(extract_loader, model, device, num_embed=512, save_path=save_path, img_size=img_size, mode=args.mode)

########################################

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset_path", default='dataset', type=str)
	parser.add_argument("--root", default='', type=str)
	parser.add_argument("--results_dir", default='results/saved_codes2', type=str)
	parser.add_argument("--dataset", default='cifar100', type=str)
	parser.add_argument("--img_size", type=int, default=32)
	parser.add_argument("--mode", help="to select compression method: support 'lzma' and 'deflate';  'original' means no compression.", type=str, default='deflate')
	parser.add_argument("--model_pt", default='results/checkpoints/VQVAE2_cifar_best.pt', type=str)
	args = parser.parse_args()
	# print(args)

	main(args)


