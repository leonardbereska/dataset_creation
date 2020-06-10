import numpy as np
import torch as tc
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class VanderPool:
    def __init__(self, mu=4.):
        self.mu = mu
        self.x0 = [tc.rand(1), tc.rand(1)]

    def derivative(self, x, t):
        (x, y) = x
        return [y, (self.mu * (1. - x * x) * y - x)]


class Lorenz:
    def __init__(self, sigma=10., beta=8. / 3, rho=28.0):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.x0 = [tc.rand(1), tc.rand(1), tc.tensor(25) + tc.rand(1)]

    def derivative(self, x, t):
        (x, y, z) = x
        return [self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]


class Lorenz96:
    def __init__(self, n_variables=36, forcing=8):
        self.n_variables = n_variables
        self.forcing = forcing
        self.x0 = self.forcing * np.ones(self.n_variables)
        self.x0[-1] += 0.01  # add small perturbation

    def derivative(self, x, t):
        N = self.n_variables
        d = np.zeros(N)
        d[0] = (x[1] - x[N - 2]) * x[N - 1] - x[0]
        d[1] = (x[2] - x[N - 1]) * x[0] - x[1]
        d[N - 1] = (x[0] - x[N - 3]) * x[N - 2] - x[N - 1]
        for i in range(2, N - 1):
            d[i] = (x[i + 1] - x[i - 2]) * x[i - 1] - x[i]
        d = d + self.forcing
        return d


def normalize(data):
    data = (data - data.mean(0)) / data.std(0)
    return data


def get_data(dynsys, time_steps=1000, step_size=0.01, noise=False):
    cut_off = 100
    time_steps = time_steps + cut_off
    t = np.arange(0.0, time_steps * step_size, step_size)
    data = odeint(dynsys.derivative, dynsys.x0, t)
    data = data[cut_off:]
    data = tc.Tensor(data)
    data = normalize(data)
    if noise:
        data += 0.01 * tc.randn(data.shape)
    return data


def plot_2d(x):
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x[:, 0], x[:, 1])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.show()


def plot_3d(x):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    plt.show()


if __name__ == '__main__':
    vdp = VanderPool(mu=4.)
    x = get_data(vdp, time_steps=1000, step_size=0.03)
    plot_2d(x)

    lor = Lorenz()
    x = get_data(lor)
    plot_3d(x)

    lor96 = Lorenz96(n_variables=36, forcing=8)
    x = get_data(lor96)
    plot_3d(x)