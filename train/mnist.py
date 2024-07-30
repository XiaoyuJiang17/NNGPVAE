import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.mnistGPVAE import MNISTGPVAE
from util import MNISTDataset

mnist_data_path = '/Users/jiangxiaoyu/Desktop/All_Projects/nearest_neighbor_GP_VAE/data/MNIST_data/train_data3.p'
train_data_dict = pickle.load(open(mnist_data_path, 'rb'))

train_data_aux_data_np = train_data_dict['aux_data']
train_data_aux_data_tensor = torch.tensor(train_data_aux_data_np)
train_data_image_np = np.transpose(train_data_dict['images'], (0, 3, 1, 2))

my_model = MNISTGPVAE(N=train_data_aux_data_np.shape[0],
                      auxi_dim=train_data_aux_data_np.shape[-1],
                      latent_dim=16)

# Create dataset and dataloader
dataset = MNISTDataset(data_X=train_data_aux_data_np, data_Y=train_data_image_np, H=5)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for idx, curr_x, curr_y, nn_idx, nn_x, nn_y in dataloader:
    nn_y_shape = nn_idx.shape
    nn_y_reshape = nn_y.view(nn_y_shape[0] * nn_y_shape[1], 1, 28, 28)
    print(my_model.encode(nn_y_reshape).shape)
    # print(f"idx {idx.shape}\n")
    # print(f"curr_x {curr_x.shape}\n")
    # print(f"curr_y {curr_y.shape}\n")
    # print(f"nn_idx {nn_idx.shape}\n")
    # print(f"nn_x {nn_x.shape}\n")
    # print(f"nn_y {nn_y.shape}\n")
    break