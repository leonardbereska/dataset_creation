import os
import torch as tc
import numpy as np
import scipy.io as io

import simulate_dynamical_system as ds


class Dataset:
    def __init__(self, data):
        """data is a tc tensor with shape (time steps, dimension)"""
        self.data = tc.FloatTensor(data)

    def add_gaussian_noise(self, noise_level):
        self.data += noise_level * tc.randn_like(self.data)

    def normalize(self):
        self.data = (self.data - self.data.mean(0)) / self.data.std(0)

    def save(self, batch_size, save_name):
        def batchify(data, batch_size):
            time_series_length, dimension = data.shape
            n_batches = int(time_series_length / batch_size)
            assert int(time_series_length / batch_size) == time_series_length / batch_size
            return data.reshape((n_batches, batch_size, dimension))

        data = self.data.numpy()
        data = batchify(data=data, batch_size=batch_size)
        print('created {} dataset of shape {}'.format(save_name, data.shape))
        save_path = os.path.join('datasets', save_name)
        np.save(save_path, data)


def create_lorenz_dataset():
    time_steps = 100000
    step_size = 0.01
    noise_level = 0.01
    batch_size = 1000
    save_name = 'lorenz_data'

    lorenz = ds.Lorenz()
    data = lorenz.simulate_system(time_steps=time_steps, step_size=step_size)
    dataset = Dataset(data)
    dataset.normalize()
    dataset.add_gaussian_noise(noise_level=noise_level)
    dataset.save(batch_size=batch_size, save_name=save_name)


def create_lorenz96_dataset():
    forcing = 8
    n_variables = 10
    step_size = 0.03
    time_steps = 100000
    noise_level = 0.01
    batch_size = 1000
    save_name = 'lorenz96_data'

    lorenz96 = ds.Lorenz96(n_variables=n_variables, forcing=forcing)
    data = lorenz96.simulate_system(time_steps=time_steps, step_size=step_size)
    dataset = Dataset(data)
    dataset.normalize()
    dataset.add_gaussian_noise(noise_level=noise_level)
    dataset.save(batch_size=batch_size, save_name=save_name)


def create_james_dataset(data_path=None):
    if data_path is None:
        data_path = '../del_alt_11_26forDaniel_KDEnonopt200000_withCatcf_cleaned.mat'
    data_mat = io.loadmat(data_path)
    data = data_mat['iFRtrafo'][0]
    trial_list = to_list(data)
    trial_list = [tc.FloatTensor(trial.T) for trial in trial_list]
    return trial_list


def to_list(data):
    return [data[i] for i in range(len(data))]


if __name__ == '__main__':
    create_lorenz_dataset()
    create_lorenz96_dataset()

    # TODO experimental data to numpy format, e.g. james
