# Created by Victoria Zhang at 8/15/2022
# File: transition_matrix.py
# Description: upload transition matrix for each video, and apply L0(spasity, entropy),
# L1/L2, and more metrics to analyze
# Scenario:
# Usage:
#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy import stats
from pathlib import Path
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
from data.load_data import load_pt_data
#%%
if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
else:
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
#%%
b_o_colors = ['#1f77b4', '#ff7f0e']
#%%
project_name = 'BD20-Feb25-2023'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(cfg['project_path'],"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
n_scores = 11
model_name = 'VAME'

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data()
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
score_bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
n_subject_in_population = len(control_videos)
#%%
titles = ["CP", "BD"]
N = [0, 0]
Labels = [[], []]
TM = [[], []]
population_TM = [[], []]
population_TM_ctl = [[],[]]
population_TM_score = [[], []]

transition_matrices = []
transition_matrices_ctl = []
transition_matrices_score = []

Epoch1_labels = [[], []]
Epoch1_transition_matrix = [[], []]
Epoch1_Entropies = [[],[]]
Epoch1_num_zero_rows = [[],[]]
Epoch1_num_zeros = [[],[]]
Epoch1_num_ones = [[],[]]
Epoch1_labels_ctl = [[], []]
Epoch1_transition_matrix_ctl = [[], []]
Epoch1_Entropies_ctl = [[],[]]
Epoch1_num_zero_rows_ctl = [[],[]]
Epoch1_num_zeros_ctl = [[],[]]
Epoch1_num_ones_ctl = [[],[]]
Epoch1_labels_score = [[], []]
Epoch1_transition_matrix_score = [[], []]
Epoch1_Entropies_score = [[],[]]
Epoch1_num_zero_rows_score = [[],[]]
Epoch1_num_zeros_score = [[],[]]
Epoch1_num_ones_score = [[],[]]

Epoch2_labels = [[], []]
Epoch2_transition_matrix = [[], []]
Epoch2_Entropies = [[],[]]
Epoch2_num_zero_rows = [[],[]]
Epoch2_num_zeros = [[],[]]
Epoch2_num_ones = [[],[]]
Epoch2_labels_ctl = [[], []]
Epoch2_transition_matrix_ctl = [[], []]
Epoch2_Entropies_ctl = [[],[]]
Epoch2_num_zero_rows_ctl = [[],[]]
Epoch2_num_zeros_ctl = [[],[]]
Epoch2_num_ones_ctl = [[],[]]
Epoch2_labels_score = [[], []]
Epoch2_transition_matrix_score = [[], []]
Epoch2_Entropies_score = [[],[]]
Epoch2_num_zero_rows_score = [[],[]]
Epoch2_num_zeros_score = [[],[]]
Epoch2_num_ones_score = [[],[]]

Epoch3_labels = [[], []]
Epoch3_transition_matrix = [[], []]
Epoch3_Entropies = [[],[]]
Epoch3_num_zero_rows = [[],[]]
Epoch3_num_zeros = [[],[]]
Epoch3_num_ones = [[],[]]
Epoch3_labels_ctl = [[], []]
Epoch3_transition_matrix_ctl = [[], []]
Epoch3_Entropies_ctl = [[],[]]
Epoch3_num_zero_rows_ctl = [[],[]]
Epoch3_num_zeros_ctl = [[],[]]
Epoch3_num_ones_ctl = [[],[]]
Epoch3_labels_score = [[], []]
Epoch3_transition_matrix_score = [[], []]
Epoch3_Entropies_score = [[],[]]
Epoch3_num_zero_rows_score = [[],[]]
Epoch3_num_zeros_score = [[],[]]
Epoch3_num_ones_score = [[],[]]

Entropies = [[],[]]
num_zero_rows = [[],[]]
num_zeros = [[],[]]
num_ones = [[],[]]
Entropies_ctl = [[],[]]
num_zero_rows_ctl = [[],[]]
num_zeros_ctl = [[],[]]
num_ones_ctl = [[],[]]
Entropies_score = [[],[]]
num_zero_rows_score = [[],[]]
num_zeros_score = [[],[]]
num_ones_score = [[],[]]
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
YMRS_score = []
HAM_D_score = []
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        YMRS_score.append(YMRS[v][0])
        HAM_D_score.append(HAM_D[v][0])
        print("Loading {}-{} data {}/{}...".format(v, titles[j], i, len(videos)))

        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v, n_cluster, n_cluster, v))
        transition_m = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(onedrive_path, project_name, v, n_cluster, v))
        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path,project_name, v,n_cluster, v))
        motif_usage = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name,v, n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")

        control_label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        control_transition = compute_transition_matrices([v], [control_label], n_cluster)

        score_label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\score_labels_{}.npy'.format(onedrive_path, project_name, v,n_cluster,v))
        score_transition = compute_transition_matrices([v], [score_label], n_cluster)

        transition = transition_m.copy()
        transition_matrices.append(transition_m)
        transition_matrices_ctl.append(control_transition)
        transition_matrices_score.append(score_transition)

        # Entropy, and classic metrics of transition matrix
        num_zero_row, num_one_item, num_zero_item = count_zeros(transition_m)
        entropy = compute_l0_entropy(transition_m, label[-1])
        Entropies[j].append(entropy)
        num_zero_row_ctl, num_one_item_ctl, num_zero_item_ctl = count_zeros(control_transition)
        entropy_ctl = compute_l0_entropy(control_transition, control_label[-1])
        Entropies_ctl[j].append(entropy_ctl)
        num_zero_row_score, num_one_item_score, num_zero_item_score = count_zeros(score_transition)
        entropy_score = compute_l0_entropy(score_transition, control_label[-1])
        Entropies_score[j].append(entropy_score)

        num_state = n_cluster
        num_zero_rows[j].append(num_zero_row)
        num_ones[j].append(num_one_item)
        num_zeros[j].append(num_zero_item)
        num_zero_rows_ctl[j].append(num_zero_row_ctl)
        num_ones_ctl[j].append(num_one_item_ctl)
        num_zeros_ctl[j].append(num_zero_item_ctl)
        num_zero_rows_score[j].append(num_zero_row_score)
        num_ones_score[j].append(num_one_item_score)
        num_zeros_score[j].append(num_zero_item_score)

        start_time = start_frame[v][0]
        five_min_frame_no = int(5 * 60 * 30)
        offset = 0#  int(door_close_time - start_time)

        epoch_1_label = label[offset:five_min_frame_no + offset]
        epoch_2_label = label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label = label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]
        epoch_1_label_ctl = control_label[offset:five_min_frame_no + offset]
        epoch_2_label_ctl = control_label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label_ctl = control_label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]
        epoch_1_label_score = score_label[offset:five_min_frame_no + offset]
        epoch_2_label_score = score_label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label_score = score_label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

        epoch_1_transition_matrix = compute_transition_matrices([v], [epoch_1_label], n_cluster)
        epoch_2_transition_matrix = compute_transition_matrices([v], [epoch_2_label], n_cluster)
        epoch_3_transition_matrix = compute_transition_matrices([v], [epoch_3_label], n_cluster)
        epoch_1_transition_matrix_ctl = compute_transition_matrices([v], [epoch_1_label_ctl], n_cluster)
        epoch_2_transition_matrix_ctl = compute_transition_matrices([v], [epoch_2_label_ctl], n_cluster)
        epoch_3_transition_matrix_ctl = compute_transition_matrices([v], [epoch_3_label_ctl], n_cluster)
        epoch_1_transition_matrix_score = compute_transition_matrices([v], [epoch_1_label_score], n_cluster)
        epoch_2_transition_matrix_score = compute_transition_matrices([v], [epoch_2_label_score], n_cluster)
        epoch_3_transition_matrix_score = compute_transition_matrices([v], [epoch_3_label_score], n_cluster)

        Epoch1_labels[j].append(epoch_1_label)
        Epoch1_transition_matrix[j].append(epoch_1_transition_matrix)
        Epoch1_labels_ctl[j].append(epoch_1_label_ctl)
        Epoch1_transition_matrix_ctl[j].append(epoch_1_transition_matrix_ctl)
        Epoch1_labels_score[j].append(epoch_1_label_score)
        Epoch1_transition_matrix_score[j].append(epoch_1_transition_matrix_score)

        Epoch2_labels[j].append(epoch_2_label)
        Epoch2_transition_matrix[j].append(epoch_2_transition_matrix)
        Epoch2_labels_ctl[j].append(epoch_2_label_ctl)
        Epoch2_transition_matrix_ctl[j].append(epoch_2_transition_matrix_ctl)
        Epoch2_labels_score[j].append(epoch_2_label_score)
        Epoch2_transition_matrix_score[j].append(epoch_2_transition_matrix_score)

        Epoch3_labels[j].append(epoch_3_label)
        Epoch3_transition_matrix[j].append(epoch_3_transition_matrix)
        Epoch3_labels_ctl[j].append(epoch_3_label_ctl)
        Epoch3_transition_matrix_ctl[j].append(epoch_3_transition_matrix_ctl)
        Epoch3_labels_score[j].append(epoch_3_label_score)
        Epoch3_transition_matrix_score[j].append(epoch_3_transition_matrix_score)

        num_zero_row, num_one_item, num_zero_item = count_zeros(epoch_1_transition_matrix[0])
        entropy = compute_l0_entropy(epoch_1_transition_matrix[0], epoch_1_label[-1])
        num_zero_row_ctl, num_one_item_ctl, num_zero_item_ctl= count_zeros(epoch_1_transition_matrix_ctl[0])
        entropy_ctl = compute_l0_entropy(epoch_1_transition_matrix_ctl[0], epoch_1_label_ctl[-1])
        num_zero_row_score, num_one_item_score, num_zero_item_score= count_zeros(epoch_1_transition_matrix_score[0])
        entropy_score = compute_l0_entropy(epoch_1_transition_matrix_score[0], epoch_1_label_score[-1])
        Epoch1_Entropies[j].append(entropy)
        Epoch1_num_zero_rows[j].append(num_zero_row)
        Epoch1_num_ones[j].append(num_one_item)
        Epoch1_num_zeros[j].append(num_zero_item)
        Epoch1_Entropies_ctl[j].append(entropy_ctl)
        Epoch1_num_zero_rows_ctl[j].append(num_zero_row_ctl)
        Epoch1_num_ones_ctl[j].append(num_one_item_ctl)
        Epoch1_num_zeros_ctl[j].append(num_zero_item_ctl)
        Epoch1_Entropies_score[j].append(entropy_score)
        Epoch1_num_zero_rows_score[j].append(num_zero_row_score)
        Epoch1_num_ones_score[j].append(num_one_item_score)
        Epoch1_num_zeros_score[j].append(num_zero_item_score)

        num_zero_row_2, num_one_item_2, num_zero_item_2 = count_zeros(epoch_2_transition_matrix[0])
        entropy_2 = compute_l0_entropy(epoch_2_transition_matrix[0], epoch_2_label[-1])
        num_zero_row_2_ctl, num_one_item_2_ctl, num_zero_item_2_ctl = count_zeros(epoch_2_transition_matrix_ctl[0])
        entropy_2_ctl = compute_l0_entropy(epoch_2_transition_matrix_ctl[0], epoch_2_label_ctl[-1])
        num_zero_row_2_score, num_one_item_2_score, num_zero_item_2_score = count_zeros(epoch_2_transition_matrix_score[0])
        entropy_2_score = compute_l0_entropy(epoch_2_transition_matrix_score[0], epoch_2_label_score[-1])
        Epoch2_Entropies[j].append(entropy_2)
        Epoch2_num_zero_rows[j].append(num_zero_row_2)
        Epoch2_num_ones[j].append(num_one_item_2)
        Epoch2_num_zeros[j].append(num_zero_item_2)
        Epoch2_Entropies_ctl[j].append(entropy_2_ctl)
        Epoch2_num_zero_rows_ctl[j].append(num_zero_row_2_ctl)
        Epoch2_num_ones_ctl[j].append(num_one_item_2_ctl)
        Epoch2_num_zeros_ctl[j].append(num_zero_item_2_ctl)
        Epoch2_Entropies_score[j].append(entropy_2_score)
        Epoch2_num_zero_rows_score[j].append(num_zero_row_2_score)
        Epoch2_num_ones_score[j].append(num_one_item_2_score)
        Epoch2_num_zeros_score[j].append(num_zero_item_2_score)

        num_zero_row_3, num_one_item_3, num_zero_item_3 = count_zeros(epoch_3_transition_matrix[0])
        entropy_3 = compute_l0_entropy(epoch_3_transition_matrix[0], epoch_3_label[-1])
        num_zero_row_3_ctl, num_one_item_3_ctl, num_zero_item_3_ctl = count_zeros(epoch_3_transition_matrix_ctl[0])
        entropy_3_ctl = compute_l0_entropy(epoch_3_transition_matrix_ctl[0], epoch_3_label_ctl[-1])
        num_zero_row_3_score, num_one_item_3_score, num_zero_item_3_score = count_zeros(epoch_3_transition_matrix_score[0])
        entropy_3_score = compute_l0_entropy(epoch_3_transition_matrix_score[0], epoch_3_label_score[-1])
        Epoch3_Entropies[j].append(entropy_3)
        Epoch3_num_zero_rows[j].append(num_zero_row_3)
        Epoch3_num_ones[j].append(num_one_item_3)
        Epoch3_num_zeros[j].append(num_zero_item_3)
        Epoch3_Entropies[j].append(entropy_3_ctl)
        Epoch3_num_zero_rows_ctl[j].append(num_zero_row_3_ctl)
        Epoch3_num_ones_ctl[j].append(num_one_item_3_ctl)
        Epoch3_num_zeros_ctl[j].append(num_zero_item_3_ctl)
        Epoch3_Entropies[j].append(entropy_3_score)
        Epoch3_num_zero_rows_score[j].append(num_zero_row_3_score)
        Epoch3_num_ones_score[j].append(num_one_item_3_score)
        Epoch3_num_zeros_score[j].append(num_zero_item_3_score)

        if i == 0:
            l = label
            l_ctl = control_label
            l_score = score_label
            # tm = transition
            # tm_ctl = control_transition
            # tm_score = score_transition
        else:
            l = np.concatenate([l,label])
            l_ctl = np.concatenate([l_ctl, control_label])
            l_score = np.concatenate([l_score, score_label])
            # tm += transition

        num_points = label.shape[0]
        n += num_points

    N[j] = n
    Labels[j] = l
    population_transition_matrix = compute_transition_matrices([titles[j]], [l], n_cluster)
    population_TM[j] = population_transition_matrix
    population_transition_matrix_ctl = compute_transition_matrices([titles[j]], [l_ctl], n_cluster)
    population_TM_ctl[j] = population_transition_matrix_ctl
    population_transition_matrix_score = compute_transition_matrices([titles[j]], [l_score], n_cluster)
    population_TM_score[j] = population_transition_matrix_score
    # TM[j] = tm/n_subject_in_population

#%%   Population-level plots
transition_group = ['','_ctl', '_score']
#%% Plot transition matrix
pwd = r'{}\Behavior_VAE_data\{}\figure\transition_matrices'.format(onedrive_path, project_name)
patient_names = control_videos + BD_videos
for i in range(len(videos)*2):
    fig, axes = plt.subplots(3, 1, figsize=(3,3))
    for j in range(3):
        transition_matrix_to_plot = eval("transition_matrices{}".format(transition_group[j]))
        im = axes[j].imshow(transition_matrix_to_plot[i], cmap='viridis')
        axes[j].set_title(patient_names[i])
        axes[j].set_xticks(np.arange(n_cluster), np.arange(n_cluster))
        axes[j].set_yticks(np.arange(n_cluster), np.arange(n_cluster))
        plt.colorbar(im, ax=axes[j])
        plt.grid(None)
    fig.show()
    if i < n_subject_in_population:
        population = 'HC'
    else:
        population = 'BD'

    fname = "{}-{}_{}_transition.png".format(population, patient_names[i], n_cluster)
    fname_pdf = "{}-{}_{}_transition.pdf".format(population, patient_names[i], n_cluster)
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%% Plot L0 measures: spasity, entropy, number of 1s, number of 0s
num_metrics = 4
metric_names = ['distribution of entropy',
                'distribution of #empty state',
                'distribution of #p(state) = 1',
                'distribution of #p(state) = 0',
                'is_BD']
lims = [[-2, 4], [-5, 15], [-4, 8], [30, 120]]
fig, axes = plt.subplots(num_metrics, figsize=(5, 10))
sns.set_style("white")
CP_idx = np.zeros(n_subject_in_population)
BD_idx = np.ones(n_subject_in_population)

for j in range(len(transition_group)):
    Entropies_to_plot = eval("Entropies{}".format(transition_group[j]))
    num_zero_rows_to_plot = eval("num_zero_rows{}".format(transition_group[j]))
    num_ones_to_plot = eval("num_ones{}".format(transition_group[j]))
    num_zeros_to_plot = eval("num_zeros{}".format(transition_group[j]))

    latent_ds = pd.DataFrame(np.concatenate((
        np.concatenate((Entropies_to_plot[0],Entropies_to_plot[1]),0).reshape(-1, 1),     # 2 x 12 list
        np.concatenate((num_zero_rows_to_plot[0],num_zero_rows_to_plot[1]),0).reshape(-1, 1),
        np.concatenate((num_ones_to_plot[0],num_ones_to_plot[1]),0).reshape(-1, 1),
        np.concatenate((num_zeros_to_plot[0],num_zeros_to_plot[1]),0).reshape(-1, 1),
        np.concatenate((CP_idx, BD_idx),0).reshape(-1, 1)), 1),
        columns=metric_names)
    for i in range(num_metrics):
        print(f"          {metric_names[i]}\n")
        sns.violinplot(x=metric_names[-1], y=metric_names[i],
                    data=latent_ds, palette="muted", ax=axes[i])
        sns.stripplot(y=metric_names[i], x=metric_names[-1], data=latent_ds,
                      color="white", edgecolor="gray", ax=axes[i])
        CP = np.asarray(latent_ds[metric_names[i]][:n_subject_in_population])
        BD = np.asarray(latent_ds[metric_names[i]][n_subject_in_population:])
        corr_HAM_D_score = scipy.stats.pearsonr(np.append(CP, BD), HAM_D_score)
        corr_YMRS_score = scipy.stats.pearsonr(np.append(CP, BD), YMRS_score)
        print("          YMARS-all: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0], corr_YMRS_score[1]))
        print("          HAM_D-all: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score[0], corr_HAM_D_score[1]))


        corr_HAM_D_score_BD = scipy.stats.pearsonr(BD, HAM_D_score[n_subject_in_population:])
        corr_YMRS_score_BD = scipy.stats.pearsonr(BD, YMRS_score[n_subject_in_population:])
        corr_HAM_D_score_CP = scipy.stats.pearsonr(CP, HAM_D_score[:n_subject_in_population])
        corr_YMRS_score_CP = scipy.stats.pearsonr(CP, YMRS_score[:n_subject_in_population])
        print("          YMARS-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_CP[0], corr_YMRS_score_CP[1]))
        print("          HAM_D-CP: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_CP[0], corr_HAM_D_score_CP[1]))
        print("          YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0], corr_YMRS_score_BD[1]))
        print("          HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_BD[0], corr_HAM_D_score_BD[1]))
        # print("corr_HAM_D_score:", corr_HAM_D_score)
        # print("corr_YMRS_score:", corr_YMRS_score)
        s = stats.ttest_ind(CP, BD, nan_policy='omit', equal_var=False)
        print("{} t-stat: {:.2f}, p-val: {:.3f}".format(metric_names[i], s.statistic, s.pvalue))
        axes[i].set_xticklabels(['CP','BD'])
        axes[i].set_title('{}, p-val: {:.3f}'.format(metric_names[i], s.pvalue))
        axes[i].set_ylim(lims[i])
        axes[i].set_xlabel('population')

    plt.suptitle("15-min")
    plt.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\transition_matrices'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "L0-measures.png"
    fname_pdf = "L0-measures.pdf"
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%%

#%% Plot L1, l2 distance
from scipy.spatial.distance import cityblock
def similarity_func(u, v):
    return 1/(1+np.linalg.norm(np.asarray(u)-np.asarray(v)))

l2_matrix = np.zeros((n_subject_in_population*2,n_subject_in_population*2))
l2_matrix_ctl = np.zeros((n_subject_in_population*2,n_subject_in_population*2))
l2_matrix_score = np.zeros((n_subject_in_population*2,n_subject_in_population*2))
# l1_matrix = np.zeros((24,24))
for i in range(n_subject_in_population*2):
    for j in range(n_subject_in_population*2):
        l2_matrix[i][j] = np.linalg.norm(transition_matrices[i]-transition_matrices[j])
        l2_matrix_ctl[i][j] = np.linalg.norm(transition_matrices_ctl[i]-transition_matrices_ctl[j])
        l2_matrix_score[i][j] = np.linalg.norm(transition_matrices_score[i]-transition_matrices_score[j])
        # l1_matrix[i][j] = cityblock(transition_matrices[i],transition_matrices[j])

for j in range(len(transition_group)):
    l2_matrix_to_plot = eval("l2_matrix{}".format(transition_group[j]))
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    im = ax.imshow(l2_matrix_to_plot)
    ax.set_title('15-min L2 distance of transition matrix')
    patient_names = control_videos + BD_videos
    ax.set_xticks(np.arange(n_subject_in_population*2), patient_names, rotation=45)
    ax.set_yticks(np.arange(n_subject_in_population*2), patient_names, rotation=45)
    plt.colorbar(im)
    fig.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\transition_matrices'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "TM-similarity_{}.png".format(transition_group[j])
    fname_pdf = "TM-similarity_{}.pdf".format(transition_group[j])
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

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
CP_idx = np.zeros(n_subject_in_population)
BD_idx = np.ones(n_subject_in_population)

for k in range(len(transition_group)):
    fig, axes = plt.subplots(num_metrics, figsize=(5, 10))
    sns.set_style('darkgrid')
    for epoch in range(1,4):
        entropy = eval("Epoch{}_Entropies".format(epoch))
        num_zero_rows = eval("Epoch{}_num_zero_rows".format(epoch))
        num_ones = eval("Epoch{}_num_ones".format(epoch))
        num_zeros = eval("Epoch{}_num_zeros".format(epoch))
        print("\nEpoch {}\n".format(epoch))
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
            CP = np.asarray(latent_ds[metric_names[i]][:n_subject_in_population])
            BD = np.asarray(latent_ds[metric_names[i]][n_subject_in_population:])

            corr_HAM_D_score = scipy.stats.pearsonr(np.append(CP, BD), HAM_D_score)
            corr_YMRS_score = scipy.stats.pearsonr(np.append(CP, BD), YMRS_score)
            print("          YMARS-all: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0], corr_YMRS_score[1]))
            print("          HAM_D-all: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score[0], corr_HAM_D_score[1]))

            corr_HAM_D_score_BD = scipy.stats.pearsonr(BD, HAM_D_score[n_subject_in_population:])
            corr_YMRS_score_BD = scipy.stats.pearsonr(BD, YMRS_score[n_subject_in_population:])
            corr_HAM_D_score_CP = scipy.stats.pearsonr(CP, HAM_D_score[:n_subject_in_population])
            corr_YMRS_score_CP = scipy.stats.pearsonr(CP, YMRS_score[:n_subject_in_population])
            print("          YMARS-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_CP[0], corr_YMRS_score_CP[1]))
            print("          HAM_D-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_CP[0], corr_HAM_D_score_CP[1]))
            print("          YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0], corr_YMRS_score_BD[1]))
            print("          HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD[0], corr_HAM_D_score_BD[1]))
            s = stats.ttest_ind(CP, BD, nan_policy='omit', equal_var=False)
            print("{} , t-stat: {:.2f}, p-val: {:.3f}".format(metric_names[i], s.statistic, s.pvalue))
            axes[i].set_xticklabels(['CP', 'BD'])
            axes[i].set_title('{}, p-val: {:.3f}'.format(metric_names[i], s.pvalue))
            axes[i].set_ylim(lims[i])
            axes[i].set_xlabel('population')
        plt.suptitle("Epoch {}-{}".format(epoch, transition_group[k]))
        fig.show()
        pwd = r'{}\Behavior_VAE_data\{}\figure\transition_matrices'.format(onedrive_path, project_name)
        Path(pwd).mkdir(exist_ok=True)
        fname = f"epoch{epoch}-L0-measures-{transition_group[k]}.png"
        fname_pdf = f"epoch{epoch}-L0-measures-{transition_group[k]}.pdf"
        fig.savefig(os.path.join(pwd, fname), transparent=True)
        fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%%
from scipy.spatial.distance import euclidean, pdist, squareform
for k in range(len(transition_group)):
    for epoch in range(1,4):
        sim_matrix = np.zeros((n_subject_in_population * 2,n_subject_in_population * 2))
        epoch_tm = eval('Epoch{}_transition_matrix{}'.format(epoch, transition_group[k]))
        epoch_tm_ = np.asarray(epoch_tm[0] + epoch_tm[1])
        epoch_label = eval('Epoch{}_labels{}'.format(epoch, transition_group[k]))
        CP_label = np.concatenate(epoch_label[0], axis=0)
        BD_label = np.concatenate(epoch_label[1], axis=0)
        CP_transition_matrix = compute_transition_matrices([titles[0]], [CP_label], n_cluster)
        BD_transition_matrix = compute_transition_matrices([titles[1]], [BD_label], n_cluster)
        for i in range(n_subject_in_population * 2):
            for j in range(n_subject_in_population * 2):
                sim_matrix[i][j] = np.linalg.norm(epoch_tm_[i]-epoch_tm_[j])


        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        im = ax.imshow(sim_matrix)
        ax.set_title('Epoch {} similarity of transition matrix {}'.format(epoch, transition_group[k]))
        patient_names = control_videos + BD_videos
        ax.set_xticks(np.arange(n_subject_in_population * 2), patient_names, rotation=45)
        ax.set_yticks(np.arange(n_subject_in_population * 2), patient_names, rotation=45)
        plt.colorbar(im)
        fig.show()
        pwd = r'{}\Behavior_VAE_data\{}\figure\transition_matrices'.format(onedrive_path, project_name)
        Path(pwd).mkdir(exist_ok=True)
        fname = f"epoch{epoch}-TM-similarity-{transition_group[k]}.png"
        fname_pdf = f"epoch{epoch}-TM-similarity-{transition_group[k]}.pdf"
        fig.savefig(os.path.join(pwd, fname), transparent=True)
        fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%%
pwd = r'{}\Behavior_VAE_data\{}\figure\transition_matrices\epoch'.format(onedrive_path, project_name)
Path(pwd).mkdir(exist_ok=True)
for k in range(len(transition_group)):
    for epoch in range(1,4):
        epoch_tm = eval('Epoch{}_transition_matrix{}'.format(epoch, transition_group[k]))
        epoch_tm_ = np.asarray(epoch_tm[0] + epoch_tm[1]).squeeze()
        for i in range(n_subject_in_population * 2):
            fig, axes = plt.subplots(1,1, figsize=(3,3))
            im = axes.imshow(epoch_tm_[i], cmap='viridis', vmin=0, vmax=1)
            plt.grid(False)
            axes.set_title(patient_names[i])
            axes.set_xticks(np.arange(n_cluster), np.arange(n_cluster))
            axes.set_yticks(np.arange(n_cluster), np.arange(n_cluster))

            plt.colorbar(im, ax=axes)
    
            fig.show()
            if i < n_subject_in_population:
                population = 'HC'
            else:
                population = 'BD'
            fname = "{}-{}_{}_transition_epoch{}-{}.png".format(population, patient_names[i], n_cluster, epoch, transition_group[k])
            fname_pdf = "{}-{}_{}_transition_epoch{}-{}.pdf".format(population, patient_names[i], n_cluster, epoch, transition_group[k])
            fig.savefig(os.path.join(pwd, fname), transparent=True)
            fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)













