import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.utils.nearest_neighbors import NNUtil
class MNISTGPVAE(nn.Module):
    """
    putting inducing points on training points; mean-field variational distributions
    """
    def __init__(self, N:int, auxi_dim:int=1, latent_dim=16, kernel:str='RBF', sigma_y=0.1):
        """
        :param latent_dim: latent dimension
        """
        super(MNISTGPVAE, self).__init__()

        self.N = N
        self.auxi_dim = auxi_dim
        self.latent_dim = latent_dim
        self.sigma_y = sigma_y

        self.mean_module = ZeroMean()
        if kernel == 'RBF':
            self.covar_module = ScaleKernel(RBFKernel(auxi_dim))

        # Architecture same as SVGP-VAE
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(in_features=8 * 2 * 2, out_features=2 * self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=128),
            nn.Unflatten(1, (8, 4, 4)),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding='same'),
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding='same'),
            nn.ELU()
        )

    def encode(self, images: Tensor, return_mean_vars=False):
        encodings = self.encoder(images)
        means = encodings[:, :self.latent_dim]
        vars = torch.exp(encodings[:, self.latent_dim:])

        if return_mean_vars:
            return means, vars

        eps = torch.randn_like(vars)
        latent_samples = means + eps * vars.sqrt()

        return latent_samples

    def decode(self, latent_samples: Tensor):
        recon_images = self.decoder(latent_samples)
        return recon_images

    def tran_elbo(self, curr_y: Tensor, nn_x: Tensor, nn_y: Tensor):
        """
        Tran et al, Sparse within sparse GP (SWSGP)

        :param curr_y: of shape (mini-batch, 1, 28, 28)
        :param nn_x: of shape (mini-batch, H, auxi_dim)
        :param nn_y: of shape (mini-batch, H, 1, 28, 28)
        """
        # data likelihood term
        latent_samples = self.encode(curr_y)
        reconstruction = self.decode(latent_samples)
        nll = self.N / curr_y.shape[0] * ((reconstruction - curr_y) ** 2).sum() / (2 * (self.sigma_y ** 2))
        nll += self.N * torch.log(torch.sqrt(2 * torch.tensor(torch.pi)) * self.sigma_y)

        # KL terms
        cov_XX = self.covar_module(nn_x) # of shape (mini-batch size, H, H)

        nn_y_shape = nn_y.shape
        nn_y_reshape = nn_y.view(nn_y_shape[0] * nn_y_shape[1], nn_y_shape[2], nn_y_shape[3], nn_y_shape[4])
        means, vars = self.encode(nn_y_reshape, return_mean_vars=True)
        means_reshape, vars_reshape = means.view(nn_y_shape[0], nn_y_shape[1], self.latent_dim), vars.view(nn_y_shape[0], nn_y_shape[1], self.latent_dim) # of shape (mini-batch, H, latent_dim)

        # nn_ids = self.nearest_neighbor_structure[indices] # (mini-batch size, H)
        # nn_ids_expanded = nn_ids.unsqueeze(-1).expand(-1, -1, self.auxi_dim) # (mini-batch size, H, auxi_dim)
        # _X = torch.gather(all_train_X, 0, nn_ids_expanded) # gather along the first dimension (dimension 0),
                                                                # of shape (mini-batch size, H, auxi_dim)
        # cov_XX = self.covar_module(_X) # of shape (mini-batch size, H, H)

        return None

    def wu_elbo(self):
        """
        Wu et al, Variational Nearest Neighbor GP (VNNGP)
        :return:
        """
        return None