import scipy.io as io
import torch as tc
import simulate_dynamical_system


def get_lorenz_data(args):
    lorenz = simulate_dynamical_system.Lorenz()
    data = simulate_dynamical_system.get_data(lorenz, time_steps=args.time_steps, step_size=0.01)
    return data


def get_lorenz96_data(n_variables, time_steps):
    forcing = 8
    lorenz96 = simulate_dynamical_system.Lorenz96(n_variables=n_variables, forcing=forcing)
    data = simulate_dynamical_system.get_data(lorenz96, time_steps=time_steps, step_size=0.03)
    return data


