import numpy as np
import scipy.io

import pickle


Category_reverse_map = {'correct_np':0,
                        'correct_lt':1,
                        'correct_rt':2,
                        'correct_delay':3,
                        'incorrect_np':4,
                        'incorrect_lt':5,
                        'incorrect_rt':6,
                        'incorrect_delay':7}

lever_press = ['correct_lt', 'correct_rt', 'incorrect_lt', 'incorrect_rt']
nose_poke = ['correct_np', 'incorrect_np']
correct_lp = ['correct_lt', 'correct_rt']

T_fellet = 2 # assuming 2s of reward


def get_nonzero_ind(cat_time_series, cat_list):
    nonzero_ind = []
    for cat in cat_list:
        nonzeros = np.nonzero(cat_time_series[Category_reverse_map[cat]])
        if nonzeros[0].size > 0:
            nonzero_ind.append(nonzeros[0])
    nonzero_ind = [ind for ind_array in nonzero_ind for ind in ind_array]
    nonzero_ind.sort()

    return nonzero_ind


def get_time_intervals(cat_time_series, cat_list):
    
    nonzero_ind_list = get_nonzero_ind(cat_time_series, cat_list)
    #print(nonzero_ind_list)
    break_points = []
    for i in range(len(nonzero_ind_list)-1):
        if nonzero_ind_list[i+1] > nonzero_ind_list[i]+1:
            break_points.append(i)
        if i+2 == len(nonzero_ind_list):
            break_points.append(i+1)
    #print(break_points)
    intervals = []
    if nonzero_ind_list:
        t0 = nonzero_ind_list[0]
        for break_point in break_points:
            interval = (t0, nonzero_ind_list[break_point])
            intervals.append(interval)
            if break_point < len(nonzero_ind_list)-1:
                t0 = nonzero_ind_list[break_point+1]
    
    return intervals


def get_light_time(trial):
    # nose-poke light input
    # on if 10s after the start of lever press
    # off when nose poke starts
    intervals_lp = get_time_intervals(james['Ccf_'][0][trial], lever_press)
    print('  time of lever press: {}'.format(intervals_lp))
    intervals_np = get_time_intervals(james['Ccf_'][0][trial], nose_poke)
    print('  time of nose poke: {}'.format(intervals_np))

    light_on_times = []
    for int_lp in intervals_lp:
        if (int_lp[0] + 5*10) < (james['Ccf_'][0][trial].shape[1]):
            t_light_on = int_lp[0] + 5*10
            light_on_times.append(t_light_on)
    print('  light on time points: {}'.format(light_on_times))

    light_off_times = []
    for int_np in intervals_np:
        if int_np[0] < (james['Ccf_'][0][trial].shape[1]):
            t_light_off = int_np[0]+5
            light_off_times.append(t_light_off)
    print('  light off time points: {}'.format(light_off_times))

    return light_on_times, light_off_times


def get_lever_time(trial):
    # lever input
    # extension (on) after nose poke
    # retraction (off) after lever press
    intervals_lp = get_time_intervals(james['Ccf_'][0][trial], lever_press)
    print('  time of lever press: {}'.format(intervals_lp))
    intervals_np = get_time_intervals(james['Ccf_'][0][trial], nose_poke)
    print('  time of nose poke: {}'.format(intervals_np))
    lever_on_times = []

    for int_np in intervals_np:
        if int_np[0] < (james['Ccf_'][0][trial].shape[1]):
            t_lever_on = int_np[0]+5
            lever_on_times.append(t_lever_on)
    print('  lever on time points: {}'.format(lever_on_times))

    lever_off_times = []
    for int_lp in intervals_lp:
        if (int_lp[0]) < (james['Ccf_'][0][trial].shape[1]):
            t_lever_off = int_lp[0] + 5
            lever_off_times.append(t_lever_off)
    print('  lever off time points: {}'.format(lever_off_times))

    return lever_on_times, lever_off_times


def get_fellet_time(trial):
    #reinforcement fellet
    # on if correct lever press (assuming last for T_fellet)
    intervals_correct_lp = get_time_intervals(james['Ccf_'][0][trial], correct_lp)
    print('  time of correct lever press: {}'.format(intervals_correct_lp))

    fellet_on_times = []
    for int_lp in intervals_correct_lp:
        if (int_lp[0]) < (james['Ccf_'][0][trial].shape[1]):
            t_fellet_on = int_lp[0] + 5
            fellet_on_times.append(t_fellet_on)
    print('  fellet on time points: {}'.format(fellet_on_times))

    fellet_off_times = []
    for t in fellet_on_times:
        if t+5*T_fellet < (james['Ccf_'][0][trial].shape[1]):
            fellet_off_times.append(t + 5*T_fellet)
        else:
            fellet_off_times.append(james['Ccf_'][0][trial].shape[1])
    print('  fellet off time points: {}'.format(fellet_off_times))

    return fellet_on_times, fellet_off_times


def gen_input_on_interval(on_times, off_times):
    
    on_intervals = []
    for on_time in on_times:
        t0 = on_time
        for ind in range(len(off_times)):
            if off_times[ind] > t0:
                t1 = off_times[ind]
                break
        on_intervals.append((t0, t1))
    
    print('  input on intervals: {}'.format(on_intervals))
    return on_intervals
    
def assign_interval(input_time_arr, on_intervals, input_type):
    '''
    input_time_arr: numpy array
    on_intervals: list of on_intervals
    input_type: string, one of ['light', 'lever', 'fellet']
    '''
    if input_type == 'light':
        for on_interval in on_intervals:
            input_time_arr[0][on_interval[0]:on_interval[1]] = 1.0
    elif input_type == 'lever':
        for on_interval in on_intervals:
            input_time_arr[1][on_interval[0]:on_interval[1]] = 1.0
    elif input_type == 'fellet':
        for on_interval in on_intervals:
            input_time_arr[2][on_interval[0]:on_interval[1]] = 1.0

    return input_time_arr


def gen_input_vectors():
    input_vecs = []
    for trial in range(james['Ccf_'][0].shape[0]-1):
        trial = trial + 1
        T = james['Ccf_'][0][trial].shape[1]
        print('trial {}: len {}'.format(trial, james['Ccf_'][0][trial].shape[1]))

        input_vec = np.zeros([3, T])
        print('initialize input vector, shape: {}'.format(input_vec.shape))
        print(' - check initial nonzero element: {}'.format(np.nonzero(input_vec)))


        light_on_times, light_off_times = get_light_time(trial)
        light_on_interals = gen_input_on_interval(
            light_on_times,
            light_off_times
            )
        input_vec = assign_interval(
            input_vec,
            light_on_interals,
            'light'
            )

        lever_on_times, lever_off_times = get_lever_time(trial)
        lever_on_interals = gen_input_on_interval(
            lever_on_times,
            lever_off_times
            )
        input_vec = assign_interval(
            input_vec,
            lever_on_interals,
            'lever'
            )

        fellet_on_times, fellet_off_times = get_fellet_time(trial)
        fellet_on_interals = gen_input_on_interval(
            fellet_on_times,
            fellet_off_times
            )
        input_vec = assign_interval(
            input_vec,
            fellet_on_interals,
            'fellet')

        print('finalize input vector, shape: {}'.format(input_vec.shape))
        print('finalize input vector, nonzero light element: {}'.format(np.nonzero(
            input_vec[0])))
        print('finalize input vector, nonzero lever element: {}'.format(np.nonzero(
            input_vec[1])))
        print('finalize input vector, nonzero fellet element: {}'.format(np.nonzero(
            input_vec[2])))

        input_vecs.append(input_vec)

    print('####################')
    print('total input vecs: {}'.format(len(input_vecs)))

    return input_vecs



if __name__ == "__main__":
    # load Jame's data
    mat_fp = '/Volumes/Fred_D/Physics/science_of_mind/theoretical_neuroscience/ZI/svae-plrnn/del_alt_11_26forDaniel_KDEnonopt200000_withCatcf_cleaned.mat'

    james = scipy.io.loadmat(mat_fp)
    print(type(james))

    input_vecs = gen_input_vectors()
    with open('/Volumes/Fred_D/Physics/science_of_mind/theoretical_neuroscience/ZI/data/James_input.p', 'wb') as fo:
        pickle.dump(input_vecs, fo)

    