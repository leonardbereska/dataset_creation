import numpy as np
import torch as tc
from scipy.integrate import odeint


class DynamicalSystem:
    def __init__(self):
        self.x0 = NotImplementedError

    def derivative(self, x, t):
        raise NotImplementedError

    def simulate_system(self, time_steps=1000, step_size=0.01):
        cut_off = 1000
        time_steps = time_steps + cut_off
        t = np.arange(0.0, time_steps * step_size, step_size)
        data = odeint(self.derivative, self.x0, t)
        data = data[cut_off:]
        return data


class VanderPool(DynamicalSystem):
    def __init__(self, mu=4.):
        super(VanderPool).__init__()
        self.mu = mu
        self.x0 = [tc.rand(1), tc.rand(1)]

    def derivative(self, x, t):
        (x, y) = x
        return [y, (self.mu * (1. - x * x) * y - x)]


class Lorenz(DynamicalSystem):
    def __init__(self, sigma=10., beta=8./3, rho=28.):
        super(Lorenz).__init__()
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.x0 = [tc.rand(1), tc.rand(1), tc.tensor(25) + tc.rand(1)]

    def derivative(self, x, t):
        (x, y, z) = x
        return [self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z]


class Lorenz96(DynamicalSystem):
    def __init__(self, n_variables=36, forcing=8):
        super(Lorenz96).__init__()
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


if __name__ == '__main__':
    import visualize
    import matplotlib.pyplot as plt

    vdp = VanderPool(mu=4.)
    data = vdp.simulate_system(step_size=0.1)
    #visualize.plot_2d(data)
    plt.plot(data)

    lor = Lorenz()
    data = lor.simulate_system()
    #visualize.plot_3d(data)
    plt.plot(data)
    plt.show()

    lor96 = Lorenz96(n_variables=36, forcing=8)
    data = lor96.simulate_system()
    visualize.plot_3d(data)
    # TODO add proper visualization for Lorenz96

