#%%
import numpy as np
import scipy
import json

def count_zeros(transition_m):
    transition = transition_m.copy()
    print(transition)
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
