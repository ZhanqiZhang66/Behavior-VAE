# Created by zhanq at 5/20/2024
# File:
# Description:
# Scenario:
# Usage
from plotting.get_paths import get_my_path

myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']
'''
Colors
'''
b_o_colors = ['#1f77b4', '#ff7f0e']

'''
Load
'''
project_name = 'BD25-HC25-final-May17-2023'
project_path = f'{onedrive_path}\Behavior_VAE_data\{project_name}'
import pandas as pd
import itertools
from multiprocessing import Pool
import numpy as np
from scipy.linalg import sqrtm
import pickle
from matplotlib import pyplot as plt
from pathlib import Path
import os
from scipy import stats
#%%
n_cluster = 10

#%%
with open(f'{project_path}/data/latent_vectors.pkl', 'rb') as f:
    df = pickle.load(f)
'''
add missing diagonal distances
'''
with open(f'{project_path}/data/Wasserstein_distances_BD_HC.pkl', 'rb') as f:
    distances_HC_BD_df = pd.DataFrame(pickle.load(f))
with open(f'{project_path}/data/Wasserstein_distances_HC_HC.pkl', 'rb') as f:
    distances_HC_HC_df = pd.DataFrame(pickle.load(f))
with open(f'{project_path}/data/Wasserstein_distances_BD_BD.pkl', 'rb') as f:
    distances_BD_BD_df = pd.DataFrame(pickle.load(f))

#%%
'''
Plot the pairwise Wasserstein distances between the latent vectors of the motifs in the HC and BD groups
'''
# distances_HC_BD_df.replace([np.inf, -np.inf], 0, inplace=True)
# distances_HC_HC_df.replace([np.inf, -np.inf], 0, inplace=True)
# distances_BD_BD_df.replace([np.inf, -np.inf], 0, inplace=True)
# Calculate the mean distances for each epoch and motif
mean_distances_HC_BD = distances_HC_BD_df.groupby(['Epoch', 'Motif'])['Distance'].mean().reset_index()
mean_distances_HC_HC = distances_HC_HC_df.groupby(['Epoch', 'Motif'])['Distance'].mean().reset_index()
mean_distances_BD_BD = distances_BD_BD_df.groupby(['Epoch', 'Motif'])['Distance'].mean().reset_index()
sem_distances_HC_BD = distances_HC_BD_df.groupby(['Epoch', 'Motif'])['Distance'].sem().reset_index()
sem_distances_HC_HC = distances_HC_HC_df.groupby(['Epoch', 'Motif'])['Distance'].sem().reset_index()
sem_distances_BD_BD = distances_BD_BD_df.groupby(['Epoch', 'Motif'])['Distance'].sem().reset_index()
#%% Plotting
x = [1, 2, 3]
x2 = [1.1, 2.1, 3.1]
x3 = [1.2, 2.2, 3.2]
cmap = plt.get_cmap('tab20')
# Iterate over each motif
for motif in range(n_cluster):
    fig, ax = plt.subplots(figsize=(3, 5))
    motif_data = mean_distances_HC_BD[mean_distances_HC_BD['Motif'] == motif]
    motif_data_2 = mean_distances_HC_HC[mean_distances_HC_HC['Motif'] == motif]
    motif_data_3 = mean_distances_BD_BD[mean_distances_BD_BD['Motif'] == motif]

    yerror = sem_distances_HC_BD[sem_distances_HC_BD['Motif'] == motif]
    yerror2 = sem_distances_HC_HC[sem_distances_HC_HC['Motif'] == motif]
    yerror3 = sem_distances_BD_BD[sem_distances_BD_BD['Motif'] == motif]

    ax.errorbar(x, motif_data['Distance'], yerr=yerror['Distance'], color=cmap(motif * 2), fmt='o', linestyle='solid',
                label='HC-BD')
    ax.errorbar(x2, motif_data_2['Distance'], yerr=yerror2['Distance'], color=cmap(motif * 2), fmt='o',
                linestyle='dashed', label='HC-HC')
    ax.errorbar(x3, motif_data_3['Distance'], yerr=yerror3['Distance'], color=cmap(motif * 2), fmt='o',
                linestyle='dotted', label='BD-BD')
    plt.grid(False)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Distance')
    ax.legend()

    ax.set_ylim([30, 80])
    plt.show()

    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_trajectory_distance'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "Motif{}-pairwise-Wasserstein.png".format(motif)
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fname_pdf = "Motif{}-pairwise-Wasserstein.pdf".format(motif)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
    for epoch in range(1, 4):

        all_data_motif = distances_HC_BD_df[(distances_HC_BD_df['Motif'] == motif) & (distances_HC_BD_df['Epoch'] == epoch)]
        all_data_motif2 = distances_HC_HC_df[
            (distances_HC_BD_df['Motif'] == motif) & (distances_HC_BD_df['Epoch'] == epoch)]
        all_data_motif3 = distances_BD_BD_df[
            (distances_HC_BD_df['Motif'] == motif) & (distances_HC_BD_df['Epoch'] == epoch)]
        s_score = stats.ttest_ind(all_data_motif['Distance'], all_data_motif2['Distance'], nan_policy='omit')
        print("Motif{} Epoch{} BD-HC, HP-HP, 2 sample t-stat: {:.2f}, p<0.05 {}, p-val: {}\n".format(motif, epoch, s_score.statistic,s_score.pvalue<0.05,
                                                                                            s_score.pvalue))
        s_score = stats.ttest_ind(all_data_motif['Distance'], all_data_motif3['Distance'], nan_policy='omit')
        print("Motif{} Epoch{} BD-HC, BD-BD, 2 sample t-stat: {:.2f}, p<0.05 {}, p-val: {}\n".format(motif, epoch, s_score.statistic,s_score.pvalue<0.05,
                                                                                          s_score.pvalue))

