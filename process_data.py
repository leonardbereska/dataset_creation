import torch as tc
import scipy.io as io

import simulate_dynamical_system as ds


class Dataset:
    def __init__(self, data):
        """data is a tc tensor with shape (time steps, dimension)"""
        self.data = data

    def add_gaussian_noise(self, noise_level):
        self.data += noise_level * tc.randn_like(self.data)

    def normalize(self):
        self.data = (self.data - self.data.mean(0)) / self.data.std(0)

    def to_tensor(self):
        self.data = tc.FloatTensor(self.data)

    def to_numpy_list(self):
        data_npy = self.data.numpy()
        batchify(data_npy, batch_size=1000)

def batchify():


def get_lorenz_data():
    lorenz = ds.Lorenz()
    data = lorenz.simulate_system(time_steps=100000, step_size=0.01)
    dataset = Dataset(data)
    dataset.normalize()
    dataset.add_gaussian_noise(noise_level=0.01)
    return data


def get_lorenz96_data(n_variables, time_steps):
    forcing = 8
    lorenz96 = ds.Lorenz96(n_variables=n_variables, forcing=forcing)
    data = lorenz96.simulate_system(time_steps=time_steps, step_size=0.03)
    return data


def get_james_data(data_path=None):
    if data_path is None:
        data_path = '../del_alt_11_26forDaniel_KDEnonopt200000_withCatcf_cleaned.mat'
    data_mat = io.loadmat(data_path)
    print(data_mat.keys())
    data = data_mat['iFRtrafo'][0]
    trial_list = []
    for i in range(len(data)):
        trial = tc.FloatTensor(data[i].T)
        trial_list.append(trial)
    return trial_list

# Dataset
# transpose
# to tensor
# to list
# batchify