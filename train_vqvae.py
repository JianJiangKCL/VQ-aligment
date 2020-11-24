import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from tqdm import tqdm
import json
from torchvision.datasets import CIFAR100
from vqvae import VQVAE
def train_AE(epoch,  loader, model, opt, device, save_path):
	loader = tqdm(loader)
	criterion = nn.MSELoss()
	latent_loss_weight = 0.5
	sample_size = 12
	mse_sum = 0
	n_sum = 0
	model.train()

	for i, (img, label) in enumerate(loader):
		opt.zero_grad()

		img = img.to(device)
		inputs = img
		out, diff = model(inputs)

		recon_loss = criterion(out, inputs)
		latent_loss = diff.mean()
		loss = recon_loss + latent_loss_weight * latent_loss
		loss.backward()
		opt.step()

		part_n = inputs.shape[0]
		part_mse_sum = recon_loss.item() * part_n

		mse_sum += part_mse_sum
		n_sum += part_n
		lr = opt.param_groups[0]["lr"]
		loader.set_description(
			(
				f"epoch: {epoch + 1};  "
				f"vq_loss: {latent_loss.item():.4f};  avg mse: {mse_sum/ n_sum:.5f}; "
				f"lr: {lr:.5f}"

			)
		)
	if epoch % 50 == 0:
		model.eval()
		sample = img[:sample_size]
		with torch.no_grad():
			out, _= model(sample)
		utils.save_image(
			torch.cat([sample, out], 0),
			save_path + f"/output/{str(epoch).zfill(5)}.png",
			nrow=sample_size,
			normalize=True,
			range=(-1, 1),
		)

		model.train()
	return mse_sum/n_sum


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

	logging.info('====>  args{} '.format(args))
	num_workers = args.num_workers

	device = "cuda"
	batch_size = args.batch_size
	dataset_path = args.dataset_path

	transform_train = transform_train_cifar

	train_ds = CIFAR100(root=dataset_path, train=True, download=True, transform=transform_train)
	obtain_indices = get_indices


	classes = [i for i in range(args.pretrain_classes)]

	print('pretrain vqvae using ', classes)
	training_idx = obtain_indices(train_ds, classes, is_training=True)

	loader = DataLoader(train_ds, batch_size=batch_size, sampler=SubsetRandomSampler(training_idx),
						num_workers=num_workers, drop_last=False)

	model = VQVAE(embed_dim=args.dim_emb, n_embed=args.n_emb).to(device)
	if args.checkpoint is not None:
		model_pt = torch.load(args.checkpoint)
		model.load_state_dict(model_pt)
	opt = optim.Adam(model.parameters(), lr=args.lr)

	best_mse = 999999
	for i in range(args.epoch):

		tmp_mse = train_AE(i, loader, model, opt, device, save_path)
		if best_mse > tmp_mse:
			best_mse = tmp_mse
			logging.info('====>  Epoch{}: best_mse {} '.format(i, best_mse))
			pt_path = os.path.join(save_path, f"checkpoints/VQVAE2_cifar_best.pt")
			torch.save(model.state_dict(), pt_path)

########################################


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--epoch", type=int, default=1400)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--batch_size", default=128, type=int)
	parser.add_argument("--dataset_path", default='dataset', help='path to cifar100, or set download=True in the program', type=str)
	parser.add_argument("--root", default='', help='the project path', type=str)
	parser.add_argument("--results_dir", default='results', type=str)
	parser.add_argument("--n_emb", type=int, help=' the size of codebook, i.e. the number of embeddings', default=512)
	parser.add_argument("--dim_emb", type=int, help='the dimension of the embedding ', default=64)
	parser.add_argument("--pretrain_classes", help='the number of classes used for pretraining VQ-VAE', type=int, default=50)
	parser.add_argument("--checkpoint", help='if there is a checkpoint, resume from the checkpoint', default=None, type=str)

	args = parser.parse_args()
	main(args)


