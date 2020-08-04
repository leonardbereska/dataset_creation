import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d(data):
    assert data.shape[1] >= 2
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(data[:, 0], data[:, 1])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.show()


def plot_3d(data):
    assert data.shape[1] >= 3
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(data[:, 0], data[:, 1], data[:, 2])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    plt.show()


def plot_all_dim(data):
    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    import numpy as np
    data = np.load('datasets/eeg_data.npy')
    plot_all_dim(data[0, :, :5])
