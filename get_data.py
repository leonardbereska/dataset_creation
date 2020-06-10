import scipy.io as io
import torch as tc
from data import dynsys


def get_lorenz_data(args, noise=True):
    lorenz = dynsys.Lorenz()
    data = dynsys.get_data(lorenz, time_steps=args.time_steps, step_size=0.01, noise=noise)
    return data


def get_lorenz96_data(n_variables, time_steps, noise=False):
    forcing = 8
    lorenz96 = dynsys.Lorenz96(n_variables=n_variables, forcing=forcing)
    data = dynsys.get_data(lorenz96, time_steps=time_steps, step_size=0.03, noise=noise)
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
        # print(trial.shape)
        trial_list.append(trial)
    return trial_list


