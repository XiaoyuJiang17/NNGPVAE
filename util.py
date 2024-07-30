import torch
from torch import Tensor
from torch.utils.data import Dataset
from gpytorch.utils.nearest_neighbors import NNUtil

class MNISTDataset(Dataset):
    def __init__(self, data_X, data_Y, H:int):
        """
        Args:
            data_X (numpy.ndarray): Numpy array of shape (N, auxi_dim), auxi_dim is the
                    number of features for nearest neighbor search.
            data_Y (numpy.ndarray): Numpy array of shape (N, 1, 28, 28)
            H: how many nearest neighbors
        """
        assert len(data_X) == len(data_Y)
        assert data_Y.shape == torch.Size((data_X.shape[0], 1, 28, 28))
        self.data_X = torch.tensor(data_X, dtype=torch.float32)
        self.data_Y = torch.tensor(data_Y, dtype=torch.float32)
        self.nn_util, self.nearest_neighbor_structure = None, None
        self.set_nearest_neighbor_structure(train_x=self.data_X, H=H)

    def set_nearest_neighbor_structure(self, train_x: Tensor, H: int):
        """
        :param train_x: of shape (N, auxi_dim), where N refers to the number of data samples, auxi_dim is the
                    number of features for nearest neighbor search.
        :param H: how many nearest neighbors.
        """
        self.nn_util = NNUtil(k=H, dim=train_x.size(-1), device=train_x.device)
        self.nearest_neighbor_structure = self.nn_util.build_sequential_nn_idx(
            train_x)  # build up sequential nearest neighbor

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, idx):
        # TODO: check how this func being called
        curr_x = self.data_X[idx]
        curr_y = self.data_Y[idx]
        nn_idx = self.nearest_neighbor_structure[idx]
        nn_x = self.data_X[nn_idx]
        nn_y = self.data_Y[nn_idx]

        return idx, curr_x, curr_y, nn_idx, nn_x, nn_y
