#%%
import numpy as np
import scipy
import json

def count_zeros(transition_m):
    transition = transition_m.copy()
    zero_rows = np.all(transition == 0, axis=1)
    zero_rows_i =  np.where(zero_rows == True)
    zero_cols = np.all(transition == 0, axis=0)
    return len(zero_rows_i[0]),  np.count_nonzero(transition == 1), np.count_nonzero(transition == 0)

def add_self_transition(transition_m, last_state):
    transition = transition_m.copy()
    zero_rows = np.all(transition == 0, axis=1)
    zero_rows_i = np.where(zero_rows == True)
    zero_cols = np.all(transition == 0, axis=0)
    zero_cols_i = np.where(zero_cols == True)

    #add self transition
    if np.sum(zero_rows) != np.sum(zero_cols):
        self_transition_i = list(set(zero_rows_i[0]) ^ set(zero_cols_i[0]))
        for idx in self_transition_i:
            if idx in set(zero_rows_i[0]):
                transition[idx][idx] = 1
    if np.sum(transition_m[last_state,:]) == 0 and np.sum(transition_m[:,last_state]) != 0:
        transition[last_state][last_state] = 1
    if zero_rows_i[0].size != 0 or zero_cols_i[0].size != 0: # when there are rows or zeros, or colums of zeros
        zeros_rows_colums_i = list(set(zero_rows_i[0]) & set(zero_cols_i[0])) # remove them
        idx_to_keep = np.ones(len(transition_m), dtype=bool)
        for i in range(len(transition_m)):
            if i in zeros_rows_colums_i:
                idx_to_keep[i] = False
        transition = transition[idx_to_keep]
        transition = transition[:, idx_to_keep]
    # len_reduced = np.sum(np.all(transition == 0, axis=1))
    # n_rows_removed = np.shape(transition_m)[0] - np.shape(transition)[0]
    # if len_reduced:
    #     transition[last_state - n_rows_removed][last_state - n_rows_removed] = 1
    return transition

def compute_l0_entropy(transition_m, last_state):
    # https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
    invertible_T = add_self_transition(transition_m, last_state)
    if len(invertible_T):
        S, U = scipy.linalg.eig(invertible_T.T)
        stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        stationary = stationary / np.sum(stationary)
        stationary = stationary.real
        H = scipy.stats.entropy(invertible_T, base=2, axis=1)
        entropy = stationary.dot(H)
    else:
        entropy = 0
    return entropy

def effective_num_states(transtion_m):
    effective_num_every_state = []
    for row in transtion_m:
        sum_p_ij = np.sum(np.square(row))
        if sum_p_ij == 0:
            effective_num_every_state.append(0)
        else:
            effective_num_every_state.append(1/sum_p_ij)
    effective_num_avg = np.mean(effective_num_every_state)
    return effective_num_every_state, effective_num_avg

#%% Retrieve and truncate motif labels
def load_motif_labels(path, videos, frames, split=1):
    labels = {}
    for v in videos:
        if split == 1:
            labels[v] = np.load(path.format(v, v))[0: frames]
        else:
            labels[v] = []
            for i in range(split):
                labels[v].append(np.load(path.format(v, v))[i * frames // split: (i + 1) * frames // split])
    return labels
# %%
def load_tmatrices(path, videos, split=1):
    matrices = {}
    for v in videos:
        matrices[v] = []
        for i in range(split):
            cpath = path.format(v, i, split)
            matrices[v].append(np.load(cpath).tolist())
    return matrices

def save_tmatrices(path, videos, matrices, split=1):
    for v in videos:
        for i in range(split):
            cpath = path.format(v, i, split)
            np.save(cpath, np.array(matrices[v][i]))


# %%
def my_colormap(coolors):
    colors = []
    if coolors == 'blue2red10':
        # continuous specturm from dark blue to red with no orange or purple
        # https://coolors.co/palette/033270-1368aa-4091c9-9dcee2-fedfd4-f29479-f26a4f-ef3c2d-cb1b16-65010c
        colors = ['#033270', '#005f73','#4091c9', '#9dcee2', '#fedfd4', '#f29479', '#f26a4f', '#ef3c2d', '#cb1b16', '#65010c']
    elif coolors == 'blackbrown10':
        # https://coolors.co/palette/797d62-9b9b7a-baa587-d9ae94-f1dca7-ffcb69-e8ac65-d08c60-b58463-997b66
        colors = ['#797d62', '#9b9b7a', '#baa587', '#d9ae94', '#f1dca7', '#ffcb69', '#e8ac65', '#d08c60', '#b58463', '#997b66']
    elif colors == 'greenorange5':
        # https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    elif coolors == 'blue2red5':
        # https://coolors.co/palette/264653-2a9d8f-e9c46a-f4a261-e76f51
        colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    elif coolors == 'greyscale_categorical6':
        # https://coolors.co/palette/3c4657-3c4e57-ced7dd-f4edea-83858c-474444
        colors = ['#ced7dd', '#83858c', '#474444','#f4edea', '#3c4e57','#3c4657']
    elif coolors == 'blue_n_orange10':
        # https://coolors.co/palette/ff5400-ff6d00-ff8500-ff9100-ff9e00-00b4d8-0096c7-0077b6-023e8a-03045e
        colors = ['#ff5400', '#ff6d00', '#ff8500', '#ff9100', '#ff9e00', '#00b4d8', '#0096c7', '#0077b6', '#023e8a', '#03045e']
    elif coolors == 'blue_n_orange2':
        # https://coolors.co/palette/ff5400-ff6d00-ff8500-ff9100-ff9e00-00b4d8-0096c7-0077b6-023e8a-03045e
        colors = ['#ff9e00', '#00b4d8']
    elif coolors == 'gold_n_purple2':
        # https://coolors.co/palette/ffe863-ffe150-ffd93d-facb2e-f5bd1f-722e9a-682a92-5d2689-522882-47297b
        colors = ['#f5bd1f', '#722e9a']
    else:
        # https://coolors.co/palette/001219-005f73-0a9396-94d2bd-e9d8a6-ee9b00-ca6702-bb3e03-ae2012-9b2226
        colors = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012', '#9b2226']
        print("Use default color scheme")
    return colors