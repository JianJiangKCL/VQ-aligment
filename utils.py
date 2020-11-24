import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision.utils import save_image, make_grid
import os
import logging.config
import matplotlib.pyplot as plt
import random

from tqdm import tqdm

import logging
from torchvision import transforms
from aug_transform import CIFAR10Policy, Cutout

transform_show = transforms.Compose([
    transforms.ToTensor()
])
transform_Img32 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_Img64 = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.4966, 0.5011, 0.5026), (0.2882, 0.2864, 0.2882))
])
transform_train_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_train_ADA_cifar=transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(p=0.5),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_recon_cifar = transforms.Compose([

    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))
])
transform_ADA_recon_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4, fill=128),
    transforms.RandomHorizontalFlip(p=0.5),
    CIFAR10Policy(),
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16),
    transforms.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))
])

from aug_transform import Cutout_latent, Flip_latent
transform_latent = transforms.Compose([
    Flip_latent(p=0.5),
    Cutout_latent(n_holes=1, length=1),
])
trans2PIL = transforms.ToPILImage(mode='RGB')

def normalize_classes4cifar(classes):

    normalized_classes = [i for i in range(100)]

    for i, cls in enumerate(classes):
        normalized_classes[cls] = i

    normalized_classes = torch.LongTensor(normalized_classes)
    # normalized_indices = normalized_classes[indices]
    return normalized_classes

def add_sn(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        return spectral_norm(m)
    else:
        return m


def reparameterize(mu, std):
    z = torch.randn_like(mu) * std + mu
    return z

def cal_dist(m, n):
    m = m.reshape(m.size(0), -1)

    n = n.reshape(n.size(0), -1).permute(1, 0)
    dist = (
        # AxD => Ax1 ; x^2
            m.pow(2).sum(1, keepdim=True)
            # AxD @ DxB => AxB ; -2x*y
            - 2 * m @ n
            # DxB => 1xB; y^2
            + n.pow(2).sum(0, keepdim=True)
    )
    return dist


def save_reconstructed_images(data, epoch, outputs, save_path, mode, expert):
    name = str(mode) + '_recon_cls' + str(expert)
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_epoch' + str(epoch) + '.png'), nrow=n, normalize=True)


def config_logging(log_file='log.txt', resume=False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode
                        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_indices(dataset, class_name, is_training):
    indices = []
    for i in range(len(dataset.targets)):
        label = dataset.targets[i]
        if label in class_name:
            indices.append(i)

    random.shuffle(indices)
    return indices


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()


def cal_cossim(input):
    '''
    input BxD; where cossim is calculated among B
    '''
    cos = nn.CosineSimilarity(dim=0)
    for i in range(int(len(input) + 1)):
        for j in range(i + 1, len(input)):
            cos_sim = cos(input[i].flatten(), input[j].flatten())
            print(i, ' vs ', j, ' cos sim', cos_sim)


