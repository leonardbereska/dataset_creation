import os
import torch as tc
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import simulate_dynamical_system as ds


class Dataset:
    def __init__(self, data):
        """data is an array (or tensor) with shape (time steps, dimension)"""
        self.data = data
        print('{} time steps, {} dimensions'.format(data.shape[0], data.shape[1]))

    def to_tensor(self):
        self.data = tc.FloatTensor(self.data)

    def show(self):
        plt.plot(self.data[:1000, :5])
        plt.show()

    def crop_shape(self, shape):
        shape_before = self.data.shape
        self.data = self.data[:shape[0], :shape[1]]
        shape_after = self.data.shape
        print('cropped from {} to {}'.format(shape_before, shape_after))

    def add_gaussian_noise(self, noise_level):
        self.data += noise_level * tc.randn_like(self.data)

    def kernel_smoothen(self, kernel_sigma=3):
        """
        Smoothen data with Gaussian kernel
        @param kernel_sigma: standard deviation of gaussian, kernel_size is adapted to that
        @return: internal data is modified but nothing returned
        """
        data = self.data

        def get_kernel(sigma):
            size = sigma * 10 + 1
            kernel = list(range(size))
            kernel = [float(k) - int(size / 2) for k in kernel]

            def gauss(x, sigma=sigma):
                return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1 / 2 * (x / sigma) ** 2)

            kernel = [gauss(k) for k in kernel]
            kernel = [k / np.sum(kernel) for k in kernel]
            return kernel

        kernel = get_kernel(kernel_sigma)
        data_final = data.copy()
        for i in range(data.shape[1]):
            data_conv = np.convolve(data[:, i], kernel)
            pad = int(len(kernel) / 2)
            data_final[:, i] = data_conv[pad:-pad]
        self.data = data_final

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
    dataset.to_tensor()
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
    dataset.to_tensor()
    dataset.normalize()
    dataset.add_gaussian_noise(noise_level=noise_level)
    dataset.save(batch_size=batch_size, save_name=save_name)


def create_james_dataset(data_path=None):
    if data_path is None:
        data_path = '../del_alt_11_26forDaniel_KDEnonopt200000_withCatcf_cleaned.mat'
    data_mat = io.loadmat(data_path)
    data = data_mat['iFRtrafo'][0]
    trial_list = [data[i] for i in range(len(data))]
    trial_list = [tc.FloatTensor(trial.T) for trial in trial_list]
    return trial_list


def create_eeg_dataset():
    data_raw = np.load('datasets/eeg_data/data_pre.npy').T
    dataset = Dataset(data_raw)
    dataset.show()
    dataset.kernel_smoothen(kernel_sigma=3)
    dataset.show()
    dataset.to_tensor()
    dataset.crop_shape((687000, 120))
    dataset.normalize()
    dataset.add_gaussian_noise(noise_level=0.01)
    dataset.save(batch_size=1000, save_name='eeg_data')


if __name__ == '__main__':
    # create_lorenz_dataset()
    # create_lorenz96_dataset()
    create_eeg_dataset()
