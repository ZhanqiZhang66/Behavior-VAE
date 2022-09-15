# Created by Victoria Zhang at 8/15/2022
# File: transition_matrix.py
# Description: upload transition matrix for each video, and apply L0(spasity, entropy),
# L1/L2, and more metrics to analyze
# Scenario:
# Usage:
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
import seaborn as sns
from scipy import stats
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
#%%
project_name = 'BD20-Jun5-2022'
config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
n_cluster = 10
model_name = 'VAME'
control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
start_frame = pd.read_csv('G:\start_frame.csv')


titles = ["CP", "BD"]
N = [0, 0]

Motif_usages = [[], []]
Motif_usage_pct = [[], []]
motif_usage_cat = [[], []]
Latent_vectors = [[], []]
Labels = [[], []]
TM = [[], []]
population_TM = [[], []]
Cluster_center = []
transition_matrices = []

Epoch1_labels = [[], []]
Epoch1_motif_usage = [[], []]
Epoch1_latent_vector = [[], []]
Epoch1_transition_matrix = [[], []]
Epoch1_motif_usage_cat = [[],[]]
Epoch1_Entropies = [[],[]]
Epoch1_num_zero_rows = [[],[]]
Epoch1_num_zeros = [[],[]]
Epoch1_num_ones = [[],[]]


Epoch2_labels = [[], []]
Epoch2_motif_usage = [[], []]
Epoch2_latent_vector = [[], []]
Epoch2_transition_matrix = [[], []]
Epoch2_motif_usage_cat = [[],[]]
Epoch2_Entropies = [[],[]]
Epoch2_num_zero_rows = [[],[]]
Epoch2_num_zeros = [[],[]]
Epoch2_num_ones = [[],[]]

Epoch3_labels = [[], []]
Epoch3_motif_usage = [[], []]
Epoch3_latent_vector = [[], []]
Epoch3_transition_matrix = [[], []]
Epoch3_motif_usage_cat = [[],[]]
Epoch3_Entropies = [[],[]]
Epoch3_num_zero_rows = [[],[]]
Epoch3_num_zeros = [[],[]]
Epoch3_num_ones = [[],[]]

Entropies = [[],[]]
num_zero_rows = [[],[]]
num_zeros = [[],[]]
num_ones = [[],[]]
#%%
def count_zeros(transition_m):
    transition = transition_m.copy()
    zero_rows = np.all(transition == 0, axis=1)
    zero_rows_i =  np.where(zero_rows == True)
    zero_cols = np.all(transition == 0, axis=0)
    return len(zero_rows_i[0]),  np.count_nonzero(transition == 1), np.count_nonzero(transition == 0)
def add_self_transition(transition_m, last_state):
    transition = transition_m.copy()
    zero_rows = np.all(transition == 0, axis=1)
    zero_rows_i =  np.where(zero_rows == True)
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

#%% Load transition matrix in each video, and append into two classes (BD, CP)
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Loading {} data...".format(v))

        label = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(project_name, v,n_cluster,n_cluster,v))
        transition_m = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(project_name, v,n_cluster, v))
        cluster_center = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(project_name, v,n_cluster, v))
        motif_usage = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 30
        transition = transition_m.copy()

        transition_matrices.append(transition_m)

        # Entropy, and classic metrics of transition matrix
        num_zero_row, num_one_item, num_zero_item = count_zeros(transition_m)
        entropy = compute_l0_entropy(transition_m, label[-1])
        Entropies[j].append(entropy)

        num_state = n_cluster
        num_zero_rows[j].append(num_zero_row)
        num_ones[j].append(num_one_item)
        num_zeros[j].append(num_zero_item)


        v_index = start_frame.loc[start_frame['video_name'] == v].index.values[0]
        door_close_time = start_frame.loc[v_index, 'door_close']
        start_time = start_frame.loc[v_index, 'n']
        five_min_frame_no = int(5 * 60 * 30)
        offset = int(door_close_time - start_time)

        epoch_1_label = label[offset:five_min_frame_no + offset]
        epoch_2_label = label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label = label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

        epoch_1_motif_usage = get_motif_usage(epoch_1_label, n_cluster)
        epoch_2_motif_usage = get_motif_usage(epoch_2_label, n_cluster)
        epoch_3_motif_usage = get_motif_usage(epoch_3_label, n_cluster)

        epoch_1_latent_vector = latent_vector[offset: five_min_frame_no + offset]
        epoch_2_latent_vector = latent_vector[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_latent_vector = latent_vector[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

        epoch_1_transition_matrix = compute_transition_matrices([v], [epoch_1_label], n_cluster)
        epoch_2_transition_matrix = compute_transition_matrices([v], [epoch_2_label], n_cluster)
        epoch_3_transition_matrix = compute_transition_matrices([v], [epoch_3_label], n_cluster)

        Epoch1_labels[j].append(epoch_1_label)
        Epoch1_motif_usage[j].append(epoch_1_motif_usage/ np.sum(epoch_1_motif_usage))
        Epoch1_latent_vector[j].append(epoch_1_latent_vector)
        Epoch1_transition_matrix[j].append(epoch_1_transition_matrix)

        Epoch2_labels[j].append(epoch_2_label)
        Epoch2_motif_usage[j].append(epoch_2_motif_usage/ np.sum(epoch_2_motif_usage))
        Epoch2_latent_vector[j].append(epoch_2_latent_vector)
        Epoch2_transition_matrix[j].append(epoch_2_transition_matrix)

        Epoch3_labels[j].append(epoch_3_label)
        Epoch3_motif_usage[j].append(epoch_3_motif_usage/ np.sum(epoch_3_motif_usage))
        Epoch3_latent_vector[j].append(epoch_3_latent_vector)
        Epoch3_transition_matrix[j].append(epoch_3_transition_matrix)

        num_zero_row, num_one_item, num_zero_item = count_zeros(epoch_1_transition_matrix[0])
        entropy = compute_l0_entropy(epoch_1_transition_matrix[0], epoch_1_label[-1])
        Epoch1_Entropies[j].append(entropy)
        Epoch1_num_zero_rows[j].append(num_zero_row)
        Epoch1_num_ones[j].append(num_one_item)
        Epoch1_num_zeros[j].append(num_zero_item)

        num_zero_row, num_one_item, num_zero_item = count_zeros(epoch_2_transition_matrix[0])
        entropy = compute_l0_entropy(epoch_2_transition_matrix[0], epoch_2_label[-1])
        Epoch2_Entropies[j].append(entropy)
        Epoch2_num_zero_rows[j].append(num_zero_row)
        Epoch2_num_ones[j].append(num_one_item)
        Epoch2_num_zeros[j].append(num_zero_item)

        num_zero_row, num_one_item, num_zero_item = count_zeros(epoch_3_transition_matrix[0])
        entropy = compute_l0_entropy(epoch_3_transition_matrix[0], epoch_3_label[-1])
        Epoch3_Entropies[j].append(entropy)
        Epoch3_num_zero_rows[j].append(num_zero_row)
        Epoch3_num_ones[j].append(num_one_item)
        Epoch3_num_zeros[j].append(num_zero_item)

        if i == 0:
            l = label
            latent = latent_vector
            m = motif_usage
            tm = transition
            Cluster_center.append(cluster_center)
            m_e1 = epoch_1_motif_usage
            m_e2 = epoch_2_motif_usage
            m_e3 = epoch_3_motif_usage
        else:
            latent = np.concatenate([latent, latent_vector])
            l = np.concatenate([l,label])
            m += motif_usage
            tm += transition
            Cluster_center.append(cluster_center)
            m_e1 += epoch_1_motif_usage
            m_e2 += epoch_2_motif_usage
            m_e3 += epoch_3_motif_usage

        motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
        num_points = latent_vector.shape[0]
        n += num_points

    Epoch1_motif_usage_cat[j] = m_e1
    Epoch2_motif_usage_cat[j] = m_e2
    Epoch3_motif_usage_cat[j] = m_e3
    N[j] = n
    Motif_usages[j] = m
    Motif_usage_pct[j] = m/n
    Latent_vectors[j] = latent
    Labels[j] = l
    population_transition_matrix = compute_transition_matrices([titles[j]], [l], n_cluster)
    population_TM[j] = population_transition_matrix
    TM[j] = tm/12
#%%   Population-level plots
#%% Plot L0 measures: spasity, entropy, number of 1s, number of 0s
num_metrics = 4
metric_names = ['distribution of entropy',
                'distribution of #empty state',
                'distribution of #p(state) = 1',
                'distribution of #p(state) = 0',
                'is_BD']
lims = [[-2, 4], [-5, 15], [-4, 8], [30, 120]]
fig, axes = plt.subplots(num_metrics, figsize=(5, 10))
sns.set_style('darkgrid')
CP_idx = np.zeros(12)
BD_idx = np.ones(12)
latent_ds = pd.DataFrame(np.concatenate((
    np.concatenate((Entropies[0],Entropies[1]),0).reshape(-1, 1),     # 2 x 12 list
    np.concatenate((num_zero_rows[0],num_zero_rows[1]),0).reshape(-1, 1),
    np.concatenate((num_ones[0],num_ones[1]),0).reshape(-1, 1),
    np.concatenate((num_zeros[0],num_zeros[1]),0).reshape(-1, 1),
    np.concatenate((CP_idx, BD_idx),0).reshape(-1, 1)), 1),
    columns=metric_names)
sns.set_style('darkgrid')
for i in range(num_metrics):
    sns.violinplot(x=metric_names[-1], y=metric_names[i],
                data=latent_ds, palette="muted", ax=axes[i])
    sns.stripplot(y=metric_names[i], x=metric_names[-1], data=latent_ds,
                  color="white", edgecolor="gray", ax=axes[i])
    CP = np.asarray(latent_ds[metric_names[i]][:12])
    BD = np.asarray(latent_ds[metric_names[i]][12:])
    s = stats.ttest_ind(CP, BD, nan_policy='omit', equal_var=False)
    print("{} t-stat: {:.2f}, p-val: {:.3f}".format(metric_names[i], s.statistic, s.pvalue))
    axes[i].set_xticklabels(['CP','BD'])
    axes[i].set_title('{}, p-val: {:.3f}'.format(metric_names[i], s.pvalue))
    axes[i].set_ylim(lims[i])
    axes[i].set_xlabel('population')

plt.suptitle("15-min")
plt.show()
#%% Plot L1, l2 distance
from scipy.spatial.distance import cityblock
def similarity_func(u, v):
    return 1/(1+np.linalg.norm(np.asarray(u)-np.asarray(v)))

l2_matrix = np.zeros((24,24))
# l1_matrix = np.zeros((24,24))
for i in range(24):
    for j in range(24):
        l2_matrix[i][j] = np.linalg.norm(transition_matrices[i]-transition_matrices[j])
        # l1_matrix[i][j] = cityblock(transition_matrices[i],transition_matrices[j])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
im = ax.imshow(l2_matrix)
ax.set_title('15-min L2 distance of transition matrix')
patient_names = control_videos + BD_videos
ax.set_xticks(np.arange(24), patient_names, rotation=45)
ax.set_yticks(np.arange(24), patient_names, rotation=45)
plt.colorbar(im)
fig.show()

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# im = ax.imshow(l1_matrix)
# ax.set_title('15-min L1 distance of transition matrix')
# patient_names = control_videos + BD_videos
# ax.set_xticks(np.arange(24), patient_names, rotation=45)
# ax.set_yticks(np.arange(24), patient_names, rotation=45)
# plt.colorbar(im)
# fig.show()
#%% inspect high similarity ones
patient_names = control_videos + BD_videos
idx = [[11, 4], [17, 12], [16, 13]]
for i in idx:
    idx_0 = i[0]
    idx_1 = i[1]
    fig, axes = plt.subplots(2, figsize=(3,5))
    im = axes[0].imshow(transition_matrices[idx_0])
    axes[0].set_title(patient_names[idx_0])
    axes[0].set_xticks(np.arange(n_cluster), np.arange(n_cluster))
    axes[0].set_yticks(np.arange(n_cluster), np.arange(n_cluster))
    plt.colorbar(im, ax=axes[0])

    im = axes[1].imshow(transition_matrices[idx_1])
    axes[1].set_title(patient_names[idx_1])
    axes[1].set_xticks(np.arange(n_cluster), np.arange(n_cluster))
    axes[1].set_yticks(np.arange(n_cluster), np.arange(n_cluster))
    plt.colorbar(im, ax=axes[1])
    fig.show()
#%% Epoch level plots

#%% violin plot
num_metrics = 4
metric_names = ['distribution of entropy',
                'distribution of #empty state',
                'distribution of #p(state) = 1',
                'distribution of #p(state) = 0',
                'is_BD']
lims = [[-2, 4], [-5, 15], [-4, 8], [30, 120]]
CP_idx = np.zeros(12)
BD_idx = np.ones(12)
for epoch in range(1,4):
    entropy = eval("Epoch{}_Entropies".format(epoch))
    num_zero_rows = eval("Epoch{}_num_zero_rows".format(epoch))
    num_ones = eval("Epoch{}_num_ones".format(epoch))
    num_zeros = eval("Epoch{}_num_zeros".format(epoch))
    print("Epoch {}".format(epoch))

    fig, axes = plt.subplots(num_metrics, figsize=(5, 10))
    sns.set_style('darkgrid')

    latent_ds = pd.DataFrame(np.concatenate((
        np.concatenate((entropy[0], entropy[1]), 0).reshape(-1, 1),
        np.concatenate((num_zero_rows[0], num_zero_rows[1]), 0).reshape(-1, 1),
        np.concatenate((num_ones[0], num_ones[1]), 0).reshape(-1, 1),
        np.concatenate((num_zeros[0], num_zeros[1]), 0).reshape(-1, 1),
        np.concatenate((CP_idx, BD_idx), 0).reshape(-1, 1)), 1),
        columns=metric_names)

    for i in range(num_metrics):
        sns.violinplot(x=metric_names[-1], y=metric_names[i],
                       data=latent_ds, palette="muted", ax=axes[i])
        sns.stripplot(y=metric_names[i], x=metric_names[-1], data=latent_ds,
                      color="white", edgecolor="gray", ax=axes[i])
        CP = np.asarray(latent_ds[metric_names[i]][:12])
        BD = np.asarray(latent_ds[metric_names[i]][12:])
        s = stats.ttest_ind(CP, BD, nan_policy='omit', equal_var=False)
        print("{} , t-stat: {:.2f}, p-val: {:.3f}".format(metric_names[i], s.statistic, s.pvalue))
        axes[i].set_xticklabels(['CP', 'BD'])
        axes[i].set_title('{}, p-val: {:.3f}'.format(metric_names[i], s.pvalue))
        axes[i].set_ylim(lims[i])
        axes[i].set_xlabel('population')
    plt.suptitle("Epoch {}".format(epoch))
    fig.show()

#%%
from scipy.spatial.distance import euclidean, pdist, squareform
for epoch in range(1,4):
    sim_matrix = np.zeros((24,24))
    epoch_tm = eval('Epoch{}_transition_matrix'.format(epoch))
    epoch_tm_ = np.asarray(epoch_tm[0] + epoch_tm[1])
    epoch_label = eval('Epoch{}_labels'.format(epoch))
    CP_label = np.concatenate(epoch_label[0], axis=0)
    BD_label = np.concatenate(epoch_label[1], axis=0)
    CP_transition_matrix = compute_transition_matrices([titles[0]], [CP_label], n_cluster)
    BD_transition_matrix = compute_transition_matrices([titles[1]], [BD_label], n_cluster)
    for i in range(24):
        for j in range(24):
            sim_matrix[i][j] = np.linalg.norm(epoch_tm_[i]-epoch_tm_[j])


    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(sim_matrix)
    ax.set_title('Epoch {} similarity of transition matrix'.format(epoch))
    patient_names = control_videos + BD_videos
    ax.set_xticks(np.arange(24), patient_names, rotation=45)
    ax.set_yticks(np.arange(24), patient_names, rotation=45)
    plt.colorbar(im)
    fig.show()
#%%
for epoch in range(1,4):
    epoch_tm = eval('Epoch{}_transition_matrix'.format(epoch))
    epoch_tm_ = np.asarray(epoch_tm[0] + epoch_tm[1]).squeeze()
    for i in range(24):
        fig, axes = plt.subplots(1,1, figsize=(3,3))
        im = axes.imshow(epoch_tm_[i])
        axes.set_title(patient_names[i])
        axes.set_xticks(np.arange(n_cluster), np.arange(n_cluster))
        axes.set_yticks(np.arange(n_cluster), np.arange(n_cluster))
        plt.colorbar(im, ax=axes)
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\transition_matrices\epoch'
        fname = "{}_{}_transition_epoch{}.png".format(patient_names[i], n_cluster, epoch)
        fig.savefig(os.path.join(pwd, fname))












