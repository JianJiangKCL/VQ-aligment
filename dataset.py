import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
import lmdb
import os
import os.path
import torch.utils.data as data

from PIL import Image
import numpy as np
import random
from torchvision import transforms
import joblib
CodeRow = namedtuple('CodeRow', ['code_t', 'code_b', 'label'])


def write(txn, key, value):
	ret = txn.put(str(key).encode('utf-8'), pickle.dumps(value), overwrite=False)
	return ret


def get(txn, key):
	value = txn.get(str(key).encode('utf-8'))
	return value


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = np.load(fo)
	return dict


trans2PIL = transforms.ToPILImage(mode='RGB')


def denormalize(x):
	x = x * torch.Tensor((0.1, 0.1, 0.1)).view(3, 1, 1)
	x = x + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
	return x


class RuntimeDataset(data.Dataset):
	trans2PIL = transforms.ToPILImage(mode='RGB')

	def __init__(self, data, transform, type='RGB'):
		self.data = data
		self.transform = transform
		self.PIL = False
		if type == 'RGB':
			self.PIL = True

	def __getitem__(self, index):
		x = self.data[index]
		if self.PIL:
			x = denormalize(x)
			x = trans2PIL(x)
		x = self.transform(x)
		return x

	def __len__(self):
		return len(self.data)


class Denormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, x):
		x = x * torch.Tensor(self.std).view(3, 1, 1)  # .cuda()
		x = x + torch.tensor(self.mean).view(3, 1, 1)  # .cuda()
		return x


transform_toPIL = transforms.Compose([
	# todo normal aug need rotate
	Denormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	transforms.ToPILImage(mode='RGB')
])


class RuntimeEvaDataset(data.Dataset):

	def __init__(self, data, transform_AUG):
		self.data = data
		self.transform_AUG = transform_AUG

	def __getitem__(self, index):
		x = self.data[index]
		x = transform_toPIL(x)
		x = self.transform_AUG(x)
		return x

	def __len__(self):
		return len(self.data)


class ImageNet_downsampled(data.Dataset):
	'''
	‘data’ - numpy array with uint8 numbers of shape samples x 3072 (32*32*3) [3*imgsize^2]. First 1024 numbers represent red               channel, next 1024 numbers green channel, last 1024 numbers represent blue channel.
	‘labels’- number representing image class, indexing starts at 1 and it uses mapping from the map_clsloc.txt     file provided in original Imagenet devkit
	‘mean’ - mean image computed over all training samples, included for convenience, usually first preprocessing step removes mean from all images.
	'''

	def __init__(self, data_folder, idx, transform=None, img_size=32
	             ):
		self.transform = transform

		# batch file idx starts from 1 to 11.
		# for idx in range(1, 11):
		if idx == -1:
			data_file = os.path.join(data_folder, 'val_data')
			d = np.load(data_file + '.npz')
		else:
			data_file = os.path.join(data_folder, 'train_data_batch_')
			d = np.load(data_file + str(idx) + '.npz')

		# d = unpickle(data_file + str(idx) + '.npz')
		x = d['data']
		y = d['labels']


		img_size2 = img_size * img_size
		# [:,R-G-B]
		x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
		# BxCxHxW
		x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
		# if idx == 1:
		self.data = x
		self.img_size = img_size
		self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC #

		self.targets = y

	def __getitem__(self, index):

		img, target = self.data[index], self.targets[index]

		img = Image.fromarray(img)
		target = torch.LongTensor([target])
		if self.transform is not None:
			img = self.transform(img)
		if self.img_size == 64:
			sample_1 = img[:, 0:32, 0:32]
			sample_2 = img[:, 32:64, 0:32]
			sample_3 = img[:, 0:32, 32:64]
			sample_4 = img[:, 32:64, 32:64]
			# 4*3 x 32x32
			img = torch.cat([sample_1,sample_2, sample_3, sample_4],dim=0)
		return img, target

	def __len__(self):
		return len(self.data)


class CodesNpzDataset(data.Dataset):

	def __init__(self, data_path, transform=None):
		self.transform = transform
		format = data_path[-2:]
		if format == 'pz':
			d = np.load(data_path)
		elif format == 'xz':
			d = joblib.load(data_path)

		self.code_t = d['code_t']
		self.code_b = d['code_b']
		self.targets = d['label']

	def __getitem__(self, index):

		code_t, code_b, target = self.code_t[index], self.code_b[index], self.targets[index]

		return torch.from_numpy(code_t), torch.from_numpy(code_b), torch.LongTensor([target])

	def __len__(self):
		return len(self.code_t)


class BufferDS(Dataset):
	def __init__(self, path):
		self.env = lmdb.open(
			path,
			max_readers=32,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False,
		)

		if not self.env:
			raise IOError('Cannot open lmdb dataset', path)

		with self.env.begin(write=False) as txn:
			self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

	def __len__(self):
		return self.length


	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			key = str(index).encode('utf-8')

			row = pickle.loads(txn.get(key))

		return row.dec_t, row.label


class CodeDS(Dataset):
	def __init__(self, path):
		self.env = lmdb.open(
			path,
			max_readers=32,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False,
		)

		if not self.env:
			raise IOError('Cannot open lmdb dataset', path)

		with self.env.begin(write=False) as txn:
			self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

	def __len__(self):
		return self.length

	def get_cls_centroid(self, ae_id):
		with self.env.begin(write=False) as txn:
			data_key = str(ae_id) + 'centroid'
			cls_data = get(txn, data_key)
			cls_data = pickle.loads(cls_data)

		return cls_data

	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			key = str(index).encode('utf-8')

			row = pickle.loads(txn.get(key))
		return torch.from_numpy(row.code_t), torch.from_numpy(row.code_b), torch.from_numpy(row.label)

