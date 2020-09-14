# Generate EEG dataset for PLRNN-MSU

## OUTPUT (trial_num, time_series, channel)


import pickle
import numpy as np

# GLOBAL VARIABLES
## path to raw s3 concat data pickle file
RAW_PATH = '/Volumes/Fred_D/Physics/science_of_mind/research/ZI/data/EEG/concat_s3.pkl' 
## path of output eeg dataset
EEG_DATASET_FP_OUT = './eeg_dataset_trunc_all_gaussian5_norm_sub30.npy'
## random seed
SEED = 121
## sampling frequency
FREQ_SAMP = 1000
## num of chunks per mode of data
NUM_CHUNK = 3
## series length for each chunk, in  sec
T_LEN = 10.0
## num of dimensions for subset
SUB_DIM = 30


def get_truc_data(raw_data_array, ind, t_len=T_LEN):

    return raw_data_array[: , ind: ind+int(t_len*FREQ_SAMP)]


# Gaussian kernel
def get_kernel(sigma=3):
    size = sigma * 10 + 1
    kernel = list(range(size))
    kernel = [float(k)-int(size/2) for k in kernel]
    
    def gauss(x, sigma=sigma):
        return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1/2*(x/sigma)**2)
    
    kernel = [gauss(k) for k in kernel]
    kernel = [k/np.sum(kernel) for k in kernel]
    return kernel


def smoothen(data, kernel):
    data_smooth = data.copy()
    for i in range(data.shape[0]):
        data_conv = np.convolve(data[i, :], kernel)
        pad = int(len(kernel)/2)
        data_smooth[i, :] = data_conv[pad:-pad]
    
    return data_smooth


def normalize(data):
    data_norm = data.copy()
    for i in range(data.shape[0]):
        data_norm[i, :] = ( data[i, :] - data[i, :].mean() ) / data[i, :].std()
    
    return data_norm


if __name__ == "__main__":
    
    # load raw data
    with open(RAW_PATH, 'rb') as fi:
        data_dicts = pickle.load(fi)

    # random seed
    np.random.seed(SEED)
    
    # randomly choose chunks of data
    eeg_dataset = []
    for k, v in data_dicts.items():
        print(k, type(v), len(v))
        if k == 'data_encmain':
            ind_arr = np.random.randint(0, 
                high=v[0].shape[0]-T_LEN*FREQ_SAMP, 
                size=NUM_CHUNK
            )
        else:
            ind_arr = np.random.randint(0, 
                high=v.shape[1]-T_LEN*FREQ_SAMP,
                size=NUM_CHUNK
            )
        
        print('random init ind: {}'.format(ind_arr))
        for ind in ind_arr:
            if k == 'data_encmain':
                trunc_arr = []
                for _arr in v:
                    _trunc_arr = _arr[ind: ind+int(T_LEN*FREQ_SAMP)]
                    trunc_arr.append(_trunc_arr)
                trunc_arr = np.array(trunc_arr)
            else:
                trunc_arr = get_truc_data(v, ind)
            
            #print('trunc arr shape: {}'.format(trunc_arr.shape))
            trunc_arr_smooth = smoothen(trunc_arr, get_kernel(5))
            trunc_arr_smooth_norm = normalize(trunc_arr_smooth)
            trunc_arr_smooth_norm = np.transpose(trunc_arr_smooth_norm)

            trunc_arr_smooth_norm_sub = trunc_arr_smooth_norm[
                :, np.random.choice(trunc_arr_smooth_norm.shape[1], SUB_DIM, replace=False)
            ]
            # non-replacement
            
            eeg_dataset.append(trunc_arr_smooth_norm_sub)
    
    eeg_dataset = np.array(eeg_dataset)
    print('output array: {}'.format(eeg_dataset.shape))
    print('sanity check if NaN in array: {}'.format(np.isnan(eeg_dataset).any()))

    np.save(EEG_DATASET_FP_OUT, eeg_dataset)



            


    # smooth data

    '''
    ## data too long, truncate to 60s
    data = {
        'EEG': get_truc_data(data_dicts['data_pre'], 1, 60)
    }
    print(data['EEG'].shape)
    '''