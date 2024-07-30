import numpy as np
import pickle
import pandas as pd

mnist_data_path = '/Users/jiangxiaoyu/Desktop/All_Projects/nearest_neighbor_GP_VAE/data/MNIST_data/train_data3.p'
train_data_dict = pickle.load(open(mnist_data_path, 'rb'))
print(train_data_dict.keys())
print(train_data_dict['images'].shape)
# df = pd.DataFrame(train_data_dict['aux_data'])
# print(df.max())

