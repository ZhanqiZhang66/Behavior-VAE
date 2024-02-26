# Created by Victoria Zhang at 4/26/2022
# File: latent_visualization.py
# Description: visualize latent variable from latent space VAME
# Scenario:
# Usage:
#%%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import scipy
import seaborn as sns
from scipy import stats
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
from data.load_data import load_pt_data
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
#%%

if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
    onedrive_path = r'D:\OneDrive - UC San Diego'

else:
    github_path = r'D:\OneDrive - UC San Diego\GitHub'

#%%
b_o_colors = ['#1f77b4', '#ff7f0e']
#%%
project_name = 'BD25-HC25-final-May17-2023'
project_path = f'D:\OneDrive - UC San Diego\Behavior_VAE_data\{project_name}'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(project_path,"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
model_name = 'VAME'

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=
                                                                       r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\data\video-information.csv')
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
score_bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
n_subject_in_population = len(control_videos)
#%%
titles = ["CP", "BD"]
N = [0, 0]


Motif_usages = [[], []]
Motif_usage_pct = [[], []]
motif_usage_cat = [[], []]
Latent_vectors = [[], []]
Latent_len = [[],[]]
Latent_len_epoch = [[[],[]], [[],[]], [[],[]]]
Labels = [[], []]
TM = [[], []]
population_TM = [[], []]
Cluster_center = []
transition_matrices = []

Latent_vectors_per_state = [[],[]] # mean and variance of state 0,1,...9 of every person's latent vector
State_vectors_per_latent_d = [[],[]] # mean and variance of latent d = 0,1,...9 of every person's state
volume_per_person = [[],[]] # trace of cov(latent vector of this person)
Stats_per_latent_per_state = [[],[]]


Epoch1_labels = [[], []]
Epoch1_latent_vector = [[], []]
Epoch1_volume_per_person = [[],[]] # trace of cov(latent vector of this person)


Epoch2_labels = [[], []]
Epoch2_latent_vector = [[], []]
Epoch2_volume_per_person = [[],[]] # trace of cov(latent vector of this person)

Epoch3_labels = [[], []]
Epoch3_latent_vector = [[], []]
Epoch3_volume_per_person = [[],[]] # trace of cov(latent vector of this person)

#%%
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Loading {} data...".format(v))

        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))

        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        motif_usage = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 30



        Latent_len[j].append(len(latent_vector))

        # variance of each person
        K = np.cov(latent_vector.T)  # covariance matrix of the 10-D latent vector
        volume = np.trace(K)
        volume_per_person[j].append(volume)

        door_close_time = start_frame[v]
        start_time = door_close_time
        five_min_frame_no = int(5 * 60 * 30)
        offset = 0 #int(start_time - door_close_time)

        epoch_1_label = label[:five_min_frame_no + offset]
        epoch_2_label = label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label = label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]
        Latent_len_epoch[0][j].append(len(epoch_1_label))
        Latent_len_epoch[1][j].append(len(epoch_2_label))
        Latent_len_epoch[2][j].append(len(epoch_3_label))

        epoch_1_motif_usage = get_motif_usage(epoch_1_label, n_cluster)
        epoch_2_motif_usage = get_motif_usage(epoch_2_label, n_cluster)
        epoch_3_motif_usage = get_motif_usage(epoch_3_label, n_cluster)

        epoch_1_latent_vector = latent_vector[:five_min_frame_no + offset]
        epoch_2_latent_vector = latent_vector[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_latent_vector = latent_vector[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]


        Epoch1_volume_per_person[j].append(np.trace(np.cov(epoch_1_latent_vector.T)))
        Epoch2_volume_per_person[j].append(np.trace(np.cov(epoch_2_latent_vector.T)))
        Epoch3_volume_per_person[j].append(np.trace(np.cov(epoch_3_latent_vector.T)))

        epoch_1_transition_matrix = compute_transition_matrices([v], [epoch_1_label], n_cluster)
        epoch_2_transition_matrix = compute_transition_matrices([v], [epoch_2_label], n_cluster)
        epoch_3_transition_matrix = compute_transition_matrices([v], [epoch_3_label], n_cluster)

        Epoch1_labels[j].append(epoch_1_label)
        Epoch1_latent_vector[j].append(epoch_1_latent_vector)

        Epoch2_labels[j].append(epoch_2_label)
        Epoch2_latent_vector[j].append(epoch_2_latent_vector)

        Epoch3_labels[j].append(epoch_3_label)
        Epoch3_latent_vector[j].append(epoch_3_latent_vector)

        if i == 0:
            l = label
            latent = latent_vector
        else:
            latent = np.concatenate([latent, latent_vector])
            l = np.concatenate([l,label])

        motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
        num_points = latent_vector.shape[0]
        n += num_points

    latent_variance_per_state = []
    stats_per_latent_per_state = np.empty((n_cluster, np.shape(latent_vector)[1], 2))  # 10 x 10
    for state_i in range(n_cluster):
        state_i_idx = label == state_i
        latent_state_i = latent_vector[state_i_idx, :]  # time of (state i) x 10
        K_per_state = np.cov(latent_state_i.T)
        latent_variance_per_state.append([np.mean(latent_state_i), np.trace(K_per_state)])
        for latent_d in range(np.shape(latent_vector)[1]):
            latent_d_state_i = latent_state_i[:, latent_d]
            stats_per_latent_per_state[state_i, latent_d, 0] = np.nanmean(latent_d_state_i)
            stats_per_latent_per_state[state_i, latent_d, 1] = np.nanvar(latent_d_state_i)

    Stats_per_latent_per_state[j].append(stats_per_latent_per_state)
    Latent_vectors_per_state[j].append(latent_variance_per_state)

    N[j] = n
    Latent_vectors[j] = latent
    Labels[j] = l
#%% see if we can decode position latent
#%% Population-wise plot
#%% plot PCA embedding for each video, all states

import numpy as np
import matplotlib.ticker as ticker
import matplotlib
titles = ["CP", "BD"]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Computing {} ...".format(v))
        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 30

        # 3D PCA
        pca = PCA(n_components=3)
        components = pca.fit_transform(latent_vector)
        components_cluster_center = pca.fit_transform(cluster_center)
        total_var = pca.explained_variance_ratio_.sum() * 100
        principalDf = pd.DataFrame(data=components)
        finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
        finalDf.columns = ['pc 1', 'pc 2', 'pc 3','target']


        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        # plot trajectory on latent

        time = np.arange(0, len(components))
        c = np.linspace(0, 1, len(components))
        ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=c, cmap=plt.get_cmap('plasma'),
                   label='trajectory', s=50)
        # make simple, bare axis lines through space:
        xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
        yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
        zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)
        ax.set_xlabel('PC 1 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[0]))
        ax.set_ylabel('PC 2 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[1]))
        ax.set_zlabel('PC 3 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[2]))

        ax.set_xlim(-55, 55)
        ax.set_ylim(-30, 55)
        ax.set_zlim(-55, 55)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.zaxis.set_major_locator(ticker.NullLocator())
        ax.grid(False)


        # plot trajectory on latent
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        t = np.arange(10)
        for g in np.unique(label):
            i = np.where(label == g)
            cmap = plt.get_cmap('tab20')
            # alpha_arr = np.arange(0.1, 1, (1 - 0.1) / len(components[i, 0]))
            # r, g, b = matplotlib.colors.to_rgb(cmap(g * 2 + j))
            # color = [(r, g, b, alpha) for alpha in alpha_arr]
            ax1.scatter(components[i, 0], components[i, 1], components[i, 2], norm=plt.Normalize(vmin=0, vmax=9),
                       color=cmap(g * 2 + 0),s=50, label='%d' % g, alpha=0.5)
        leg =  ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # make simple, bare axis lines through space:
        xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
        ax1.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
        yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
        ax1.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
        zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
        ax1.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        ax1.set_title("PCs of {}-{}\n Exp_Var:{:.2f}".format(titles[j], v, total_var))
        ax1.set_xlabel('PC 1 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[0]))
        ax1.set_ylabel('PC 2 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[1]))
        ax1.set_zlabel('PC 3 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[2]))

        ax1.set_xlim(-55, 55)
        ax1.set_ylim(-30, 55)
        ax1.set_zlim(-55, 55)
        ax1.grid(False)
        ax1.xaxis.set_major_locator(ticker.NullLocator())
        ax1.yaxis.set_major_locator(ticker.NullLocator())
        ax1.zaxis.set_major_locator(ticker.NullLocator())




        fig.show()

        pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\Subjects-All-Motifs'.format(onedrive_path, project_name)
        Path(pwd).mkdir(parents=True, exist_ok=True)
        fname = "PCs of {}-{}-3d.png".format(titles[j], v)
        fname_pdf = "PCs of {}-{}-3d.pdf".format(titles[j], v)
        fig.savefig(os.path.join(pwd, fname), transparent=True)
        fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
        #fig.savefig(os.path.join(pwd, fname), tranparent=True)

        # # 2D PCA
        # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # for g in np.unique(label):
        #     i = np.where(label == g)
        #     cmap = plt.cm.Spectral
        #     ax.scatter(components[i, 0], components[i, 1], norm=plt.Normalize(vmin=0, vmax=9),
        #                   c=label[i], cmap='Spectral', s=2, alpha=0.05, label='%d' % g)
        #
        # ax.set_title("PCs of {}-{}\n Exp_Var:{:.2f}".format(titles[j], v, pca.explained_variance_ratio_[:-2].sum() * 100))
        # ax.set_xlabel('PC 1')
        # ax.set_ylabel('PC 2')
        # ax.set_xlim(-55, 55)
        # ax.set_ylim(-55, 55)
        # ax.set_aspect('equal', 'box')
        #
        # fig.show()
        # pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
        # fname = "PCs of {}-{}-2d.png".format(titles[j], v)
        # fig.savefig(os.path.join(pwd, fname))
#%% Plot PCA of BD and CP population, for all state
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
fig_pca = plt.figure(figsize=(10,20))
fig_latent = plt.figure(figsize=(30,30))

pca = PCA(n_components=3)
K_var = np.zeros((10, 2))
K_var_all_subjects = np.zeros((n_subject_in_population, 10, 2))

latent_vector_stack = np.vstack(Latent_vectors)
labels_stack = np.hstack(Labels)
state_volume = []
len_latent = [0, len(Latent_vectors[0])]
components = pca.fit_transform(latent_vector_stack)
for g in range(n_cluster):
    idx = np.where(labels_stack == g)[0]
    latent_this_state = latent_vector_stack[idx, :]
    K = np.cov(latent_this_state.T) # 10 x
    volume_of_group = np.trace(K)
    state_volume.append(volume_of_group)

for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    latent_vec = Latent_vectors[j]
    latent_vec_trim = latent_vec
    label = Labels[j]
    label_trim = Labels[j]
    ax = fig_pca.add_subplot(2, 1, j + 1, projection='3d')

    total_var = pca.explained_variance_ratio_.sum() * 100
    t = np.arange(10)
    for g in np.unique(label):
        i = np.where(label == g)
        ii = i[0] + len_latent[j]
        cmap = plt.get_cmap('tab20')
        ax.scatter(components[ii, 0], components[ii, 1], components[ii, 2], norm=plt.Normalize(vmin=0, vmax=9),
                   color=cmap(g * 2 + 0),
                   s=50, alpha=0.5, label='%d' % g)
    leg = ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_title("PCs of {}-\n Exp_Var:{:.2f}".format(titles[j],  total_var))
    ax.set_xlabel('PC 1 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[0]))
    ax.set_ylabel('PC 2 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[1]))
    ax.set_zlabel('PC 3 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[2]))

    ax.set_xlim(-55, 55)
    ax.set_ylim(-30, 55)
    ax.set_zlim(-55, 55)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.zaxis.set_major_locator(ticker.NullLocator())
    ax.grid(False)
    # make simple, bare axis lines through space:
    # xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
    # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
    # yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
    # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
    # zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
    # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
fig_pca.show()

pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual'.format(onedrive_path, project_name)
Path(pwd).mkdir(exist_ok=True)
fname = "PCs-of-BD-CP-3d.png"
fname_pdf = "PCs-of-BD-CP-3d.pdf"
fig_pca.savefig(os.path.join(pwd, fname), transparent=True)
fig_pca.savefig(os.path.join(pwd, fname_pdf), transparent=True)


# fname = "PCs-of-BD-CP.png"
# fig_pca.savefig(os.path.join(pwd, fname), transparent=True)
# fname1 = "PCs-of-BD-CP.pdf"
# fig_pca.savefig(os.path.join(pwd, fname1), transparent=True)
#%% Plot PCA of BD and CP population, for all state 2d
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
fig_pca = plt.figure(figsize=(10,20))
fig_latent = plt.figure(figsize=(30,30))

pca = PCA(n_components=2)
K_var = np.zeros((10, 2))
K_var_all_subjects = np.zeros((n_subject_in_population, 10, 2))

latent_vector_stack = np.vstack(Latent_vectors)
labels_stack = np.hstack(Labels)
state_volume = []
len_latent = [0, len(Latent_vectors[0])]
components = pca.fit_transform(latent_vector_stack)
for g in range(n_cluster):
    idx = np.where(labels_stack == g)[0]
    latent_this_state = latent_vector_stack[idx, :]
    K = np.cov(latent_this_state.T) # 10 x
    volume_of_group = np.trace(K)
    state_volume.append(volume_of_group)

for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    latent_vec = Latent_vectors[j]
    latent_vec_trim = latent_vec
    label = Labels[j]
    label_trim = Labels[j]
    ax = fig_pca.add_subplot(2, 1, j + 1)

    total_var = pca.explained_variance_ratio_.sum() * 100
    t = np.arange(10)
    for g in np.unique(label):
        i = np.where(label == g)
        ii = i[0] + len_latent[j]
        cmap = plt.get_cmap('tab20')
        ax.scatter(components[ii, 0], components[ii, 1], norm=plt.Normalize(vmin=0, vmax=9),
                   color=cmap(g * 2 + 0),
                   s=50, alpha=0.5, label='%d' % g)
    leg = ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_title("PCs of {}-\n Exp_Var:{:.2f}".format(titles[j],  total_var))
    ax.set_xlabel('PC 1 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[0]))
    ax.set_ylabel('PC 2 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[1]))


    ax.set_xlim(-70, 80)
    ax.set_ylim(-70, 80)

    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())

    ax.grid(False)
    # make simple, bare axis lines through space:
    # xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
    # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
    # yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
    # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
    # zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
    # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
fig_pca.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual'.format(onedrive_path, project_name)
Path(pwd).mkdir(exist_ok=True)
fname = "PCs-of-BD-CP-2d.png"
fname_pdf = "PCs-of-BD-CP-2d.pdf"
fig_pca.savefig(os.path.join(pwd, fname), transparent=True)
fig_pca.savefig(os.path.join(pwd, fname_pdf), transparent=True)

#%% Plot PCA of BD and CP population, for each state
cmap = plt.get_cmap('tab20')
titles = ["HP", "BD"]

pca = PCA(n_components=3)
K_var = np.zeros((10, 2))

len_latent = [0, len(Latent_vectors[0])]
components = pca.fit_transform(latent_vector_stack)
for g in np.unique(label):
    fig_pca = plt.figure(figsize=(10, 10))
    ax = fig_pca.add_subplot(1, 1, 1, projection='3d')
    latent_this_state = []
    for j, videos in enumerate([control_videos, BD_videos]):

        n = N[j]
        latent_vec = Latent_vectors[j]

        latent_vec_trim = latent_vec
        label = Labels[j]
        label_trim = Labels[j]

        t = np.arange(10)

        i = np.where(label == g)
        ii = i[0] + len_latent[j]
        cmap = plt.get_cmap('tab20')
        ax.scatter3D(components[ii, 0], components[ii, 1], components[ii, 2], norm=plt.Normalize(vmin=0, vmax=9),
                   color=cmap(g * 2 + j),
                   s=30, alpha=0.05,  label=titles[j])
    leg = ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    ax.set_title("PCs of state-{}-\n Exp_Var:{:.2f}".format(g,  total_var))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-30, 30)
    ax.set_zlim(-40, 40)

    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)
    ax.grid(False)
    # make simple, bare axis lines through space:
    xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
    yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
    zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
    fig_pca.show()


    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\Population-{}-motifs'.format(onedrive_path, project_name, n_cluster)
    Path(pwd).mkdir(exist_ok=True)
    fname = "PCs-of-BD-CP-STATE-{}.png".format(g)
    fname_pdf = "PCs-of-BD-CP-STATE-{}.pdf".format(g)
    fig_pca.savefig(os.path.join(pwd, fname), transparent=True)
    fig_pca.savefig(os.path.join(pwd, fname_pdf), transparent=True)


#%% Plot PCA of BD and CP population, for each state, and each subject
K_var_all_subjects = np.zeros((n_subject_in_population, 10, 2))
len_latent = [0, len(Latent_vectors[0])]
components = pca.fit_transform(latent_vector_stack)
for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    latent_vec = Latent_vectors[j]
    latent_vec_this_population = latent_vec

    # embed on PC space with all points first
    total_var = pca.explained_variance_ratio_.sum() * 100
    principalDf = pd.DataFrame(data=components)
    finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
    finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']

    label = Labels[j]
    label_this_population = label
    # now find index of this subject, and this state from the long latent vector
    previous_subject_latent_end = 0
    for sub in range(n_subject_in_population):

        if sub > 0:
            previous_subject_latent_end += Latent_len[j][sub - 1]
        this_subject_latent_len = Latent_len[j][sub]
        latent_vec_sub = latent_vec_this_population[previous_subject_latent_end: previous_subject_latent_end + this_subject_latent_len]

        label_sub = label_this_population[previous_subject_latent_end: previous_subject_latent_end + this_subject_latent_len]
        sub_name = videos[sub]
        for g in np.unique(label):
            #  state-population-subject
            #  find the index of this subject, and in this state g

            i_sub_ = np.where(label_sub == g)
            i_sub = i_sub_[0] + previous_subject_latent_end + len_latent[j]
            if len(i_sub):
                latent_sub_g = latent_vec_sub[i_sub_]
                if latent_sub_g.shape[0] < 3:
                    continue
                K_sub = np.cov(latent_sub_g.T)
                volume_of_group_sub = np.trace(K_sub)
                K_var_all_subjects[sub][g][j] = volume_of_group_sub
            else:
                volume_of_group_sub = 0
                K_var_all_subjects[sub][g][j] = volume_of_group_sub
            fig_pca_per_state = plt.figure(figsize=(10, 10))
            ax2 = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
            ax2.scatter(components[i_sub, 0], components[i_sub, 1], components[i_sub, 2], norm=plt.Normalize(vmin=0, vmax=9),
                        color=cmap(g * 2 + j), s=30, alpha=0.1, label='%d' % g)

            # make simple, bare axis lines through space:
            xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
            ax2.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
            yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
            ax2.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
            zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
            ax2.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
            for lh in leg.legendHandles:
                lh.set_alpha(1)
            ax2.set_title("PCs of {}-{}-State- {}\n Exp_Var:{:.2f}".format(titles[j], sub_name, g, total_var))
            # ax2.set_xlabel('PC 1 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[0]))
            # ax2.set_ylabel('PC 2 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[1]))
            # ax2.set_zlabel('PC 3 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[2]))
            ax2.set_xlim(-50, 50)
            ax2.set_ylim(-30, 50)
            ax2.set_zlim(-50, 50)
            fig_pca_per_state.show()

            pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\Subject-{}-Motifs'.format(onedrive_path, project_name, n_cluster)
            Path(pwd).mkdir(exist_ok=True)
            fname = "PCs of {}-{} State {}.png".format(titles[j], sub_name, g)
            fname_pdf = "PCs of {}-{} State {}.pdf".format(titles[j], sub_name, g)
            fig_pca_per_state.savefig(os.path.join(pwd, fname), transparent=True)
            fig_pca_per_state.savefig(os.path.join(pwd, fname_pdf), transparent=True)
            plt.close('all')




            #  state-population wise
            # i = np.where(label == g)
            # latent_g = latent_vec[i]
            # components = pca.fit_transform(latent_g)
            # total_var = pca.explained_variance_ratio_.sum() * 100
            # principalDf = pd.DataFrame(data=components)
            # finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
            # finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
            # K = np.cov(latent_g.T)
            # volume_of_group = np.trace(K)
            # K_var[g][j] = volume_of_group
            # fig_pca_per_state = plt.figure(figsize=(10, 10))
            # ax2 = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
            #
            # ax.scatter(components[:, 0], components[:, 1], components[:, 2],norm=plt.Normalize(vmin=0, vmax=9),
            #                   color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
            # ax2.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
            #            color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
            # ax2.set_title("PCs of {}-State {}\n Exp_Var:{:.2f}".format(titles[j], g, total_var))
            # ax2.set_xlabel('PC 1')
            # ax2.set_ylabel('PC 2')
            # ax2.set_zlabel('PC 3')
            # ax2.set_xlim(-50, 50)
            # ax2.set_ylim(-50, 50)
            # ax2.set_zlim(-50, 50)
            # fig_pca_per_state.show()
            # pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
            # fname = "PCs of {} State {}.png".format(titles[j], g)
            #fig_pca_per_state.savefig(os.path.join(pwd, fname))

    # ax.set_title("PCs of {}-\n Exp_Var:{:.2f}".format(titles[j], total_var))
    # ax.set_xlabel('PC 1')
    # ax.set_ylabel('PC 2')
    # ax.set_zlabel('PC 3')
    # ax.set_xlim(-50, 50)
    # ax.set_ylim(-50, 50)
    # ax.set_zlim(-50, 50)
    # # leg = ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    # fig_pca.show()
    # pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
    # fname = "PCs of BD-CP.png"
    #fig_pca.savefig(os.path.join(pwd, fname))
    #plt.close('all')
#
#%%

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
x = np.arange(10)
ax.plot(x, K_var[:,0], '-o')
ax.plot(x, K_var[:, 1],'-o')
ax.set_title('Volume of motif')
ax.set_xticks(x)
ax.set_xlabel('motif')
ax.set_ylabel('volume of state')
ax.set_ylim([-500, 2000])
ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
fig.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch'.format(onedrive_path, project_name)
Path(pwd).mkdir(exist_ok=True)
fname = "Volume of state {}.png".format(g)
fig.savefig(os.path.join(pwd, fname))
fname_pdf = "Volume of state {}.pdf".format(g)
fig.savefig(os.path.join(pwd, fname))
#%%
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots(figsize=(10, 10))
s = ax.scatter([], [])
x = np.arange(10)
ax.set_xlim(0,10)
ax.set_ylim(400, 1500)
ax.set_xticks(x)
ax.set_xlabel('State')
ax.set_ylabel('volume of state')
def animation(i):
    x = np.arange(10)
    for j, videos in enumerate([control_videos, BD_videos]):
        color = 'C{}'.format(j)
        for sub in range(12):
            y = K_var_all_subjects[sub,:,j]
            sub_name = videos[sub]
            s.set_offsets(np.column_stack(x[i]+j*0.2, y[i]))
            s.set_facecolor(color)
            ax.set_title('{}-{} Volume of state'.format(titles[j], sub_name))


ani = FuncAnimation(fig, animation, frames=240, interval = 500, blit = True)

plt.show()
#%%
cmap = plt.get_cmap('tab20')
for j, videos in enumerate([control_videos, BD_videos]):
    color = 'C{}'.format(j)
    for sub in range(12):
        sub_name = videos[sub]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.arange(10)
        ax.scatter(x, K_var_all_subjects[sub,:,j], norm=plt.Normalize(vmin=0, vmax=9), c=cmap(list(range(j,20,2))))
        ax.set_title('{} Volume of state'.format(sub_name))
        ax.set_xticks(x)
        ax.set_xlabel('State')
        ax.set_ylim(-500, 2000)
        ax.set_ylabel('volume of state')
        ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
        fig.show()
        pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\volume'.format(onedrive_path, project_name)

        Path(pwd).mkdir(exist_ok=True, parents=True)
        fname = "{}_{}_volume.png".format(sub_name, n_cluster)
        fig.savefig(os.path.join(pwd, fname),transparent=True)
        fname_pdf = "{}_{}_volume.pdf".format(sub_name, n_cluster)
        fig.savefig(os.path.join(pwd, fname_pdf),transparent=True)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
x = np.arange(10)
for sub in range(12):
    ax.scatter(x, K_var_all_subjects[sub, :, 0], c=b_o_colors[0], alpha=0.5)
    ax.scatter(x+0.2, K_var_all_subjects[sub, :, 1], c=b_o_colors[1], alpha=0.5)
    x = np.arange(10)

ax.plot(x, np.mean(K_var_all_subjects[:, :, 0],axis=0), '-^', color=b_o_colors[0])
ax.plot(x, np.mean(K_var_all_subjects[:, :, 1],axis=0),'-^', color=b_o_colors[1])
ax.set_title('Volume of state')
ax.set_xticks(x)
ax.set_xlabel('State')
ax.set_ylabel('volume of state')
ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
fig.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\volume'.format(onedrive_path, project_name)
fname = "{}_volume-mean.png".format(n_cluster)
fname_pdf = "{}_volume-mean.pdf".format(n_cluster)
fig.savefig(os.path.join(pwd, fname),transparent=True)
fig.savefig(os.path.join(pwd, fname_pdf),transparent=True)
#%% Epoch-wise plot


#%% Helper function
import numpy as np

# https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-a-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
#%%   PCs of each epoch, showing distance between epochs

# First, Get the centroids of each epoch of each population
centroids = [] #epoch 1 control epoch 1 hp state 0, #epoch 1 control epoch 1 hp state 1, ....epoch 2 control epoch 2 bd, ...
centroids_se = []
latent_all = [] # epoch 1 control, epoch 1 hp, epoch 2 control, epoch 2 bd, ...
label_all = []
counter = 0
for epoch in range(1, 4):
    # for each population, get the centroid of each epoch
    for j, videos in enumerate([control_videos, BD_videos]):

        latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
        label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)

        latent_all.append(latent_vec_epoch_pop) # (6, 225000, 10)
        label_all.append(label_epoch_pop) # (6, 225000)

        for g in range(n_cluster):
            idx_g = np.where(label_epoch_pop == g)[0]
            centroid = np.mean(latent_vec_epoch_pop[idx_g, :], axis=0)
            centroids.append(centroid)
        counter += 1

subject_end = [[0],[0]]
subject_latent_len = [[],[]]

latent_all_ = np.vstack(latent_all) # (1350000, 10)
label_all_ = np.hstack(label_all) # (1350000, 1)
centroids = np.squeeze(np.array(centroids)) # (60 x 10)

# Next, I need to get the embedding of all latent in all three epochs
components_pop = pca.fit_transform(np.vstack((latent_all_, centroids))) # (1350060, 10)
labels_pop = list(label_all_)
centroids_labels = [-1] * 60
labels_pop += centroids_labels

colors = ['k', 'r']
# for 10 states, plot the distance between epochs, within and between groups
state_epoch_centroids = []
state_epoch_volume = []
len_latent_epoch = [0, 5*30*60, 10*30*60, 15*30*60]
len_latent = [0, len(Latent_vectors[0])]

for g in range(10):
    fig_pca_per_state = plt.figure(figsize=(15, 10))
    ax = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
    epoch_centroids = [[],[]] # 2 x 3
    epoch_volume = [[], []]
    markers = ['o', '^', 'D']
    alphas = [0.4, 0.7, 1]
    count = 0
    for epoch in range(1, 4):
        # for each population, get the centroid of each epoch
        for j, videos in enumerate([control_videos, BD_videos]):
            latent_vec = latent_all[count]  # (1, 225000, 10)
            label_vec = label_all[count] # (1, 225000)



            latent_vec_this_population = latent_vec
            # Then, I need to get the index of state g in label_vec
            idx = np.where(label_vec == g)[0]
            latent_vec_g = latent_vec[idx, :]
            print('latent_vec_g shape: {}'.format(latent_vec_g.shape))

            # Now need to find these idx  in the (1350000, 10) array of latent_vec,
            # which is how many len_latent_epoch[225000] is away from the start
            idx_g = len_latent_each_epoch_each_population * count + idx
            count += 1

            centroid_idx = n_cluster * (j + (epoch-1)*2) + g
            transformed_centroid = components_pop[centroid_idx-60, :]
            epoch_centroids[j].append(transformed_centroid) # centroid of HP epoch 1, centroid of BD epoch 1, ...

            principalDf_pop = pd.DataFrame(data=components_pop)
            principalDf_pop_labels = pd.DataFrame(data=labels_pop)
            finalDf = pd.concat([principalDf_pop, principalDf_pop_labels], axis=1)
            finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
            if len(latent_vec_g):
                KK = np.cov(latent_vec_g.T)
                volume_of_group = np.trace(KK)
                epoch_volume[j].append(volume_of_group)
                # Plot latent swarms
                ax.plot3D(components_pop[idx_g, 0], components_pop[idx_g, 1], components_pop[idx_g, 2],
                          # norm=plt.Normalize(vmin=0, vmax=9),
                           color=cmap(g * 2 + j),
                            # s=5,
                             alpha=0.9,
                             label='%d' % g,linewidth=10, zorder=-1)
            # plot centroid
            ax.scatter3D(transformed_centroid[0], transformed_centroid[1], transformed_centroid[2],
                       norm=plt.Normalize(vmin=0, vmax=9),
                       color=colors[j],
                       marker='.',
                       s=200,
                       label='{} epoch{} centroid'.format(titles[j], epoch),zorder=1)
        # plot distance between centroids between groups in same epoch
        xs = [epoch_centroids[0][epoch - 1][0], epoch_centroids[1][epoch - 1][0]]
        ys = [epoch_centroids[0][epoch - 1][1], epoch_centroids[1][epoch - 1][1]]
        zs = [epoch_centroids[0][epoch - 1][2], epoch_centroids[1][epoch - 1][2]]
        ax.plot3D(xs, ys, zs, linewidth=3, linestyle='--',
                color='m', label='epoch{}centroid between BD-HP'.format(epoch),zorder=1)

    state_epoch_centroids.append(epoch_centroids)
    state_epoch_volume.append(epoch_volume)

    # plot distance between centroids within groups over time
    xs = [epoch_centroids[0][0][0], epoch_centroids[0][1][0]]
    ys = [epoch_centroids[0][0][1], epoch_centroids[0][1][1]]
    zs = [epoch_centroids[0][0][2], epoch_centroids[0][1][2]]
    a = Arrow3D(xs, ys,
                zs, mutation_scale=20,
                lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(a)
    # ax.plot3D(xs, ys, zs, linestyle='-', linewidth=5, color='k',
    #         label='HP centroid between epoch 1 and 2',zorder=1)
    xs = [epoch_centroids[0][1][0], epoch_centroids[0][2][0]]
    ys = [epoch_centroids[0][1][1], epoch_centroids[0][2][1]]
    zs = [epoch_centroids[0][1][2], epoch_centroids[0][2][2]]
    a = Arrow3D(xs, ys,
                zs, mutation_scale=20,
                lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(a)

    xs = [epoch_centroids[1][0][0], epoch_centroids[1][1][0]]
    ys = [epoch_centroids[1][0][1], epoch_centroids[1][1][1]]
    zs = [epoch_centroids[1][0][2], epoch_centroids[1][1][2]]
    a = Arrow3D(xs, ys,
                zs, mutation_scale=20,
                lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
    xs = [epoch_centroids[1][1][0], epoch_centroids[1][2][0]]
    ys = [epoch_centroids[1][1][1], epoch_centroids[1][2][1]]
    zs = [epoch_centroids[1][1][2], epoch_centroids[1][2][2]]
    a = Arrow3D(xs, ys,
                zs, mutation_scale=20,
                lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)

    ax.set_title("PCs of {}-State-{} \n".format(titles[j], g))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    # make simple, bare axis lines through space:
    # xAxisLine = ((np.min(components_pop[:, 0]), np.max(components_pop[:, 0])), (0, 0), (0, 0))
    # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
    # yAxisLine = ((0, 0), (np.min(components_pop[:, 1]), np.max(components_pop[:, 1])), (0, 0))
    # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
    # zAxisLine = ((0, 0), (0, 0), (np.min(components_pop[:, 2]), np.max(components_pop[:, 2])))
    # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-30, 30)
    ax.set_zlim(-40, 40)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(left=False, right=False, labelleft=False,
                   labelbottom=False, bottom=False)
    fig_pca_per_state.show()



    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_centroid'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "PCs of {} State {}-centroid.png".format(titles[j], g)
    fig_pca_per_state.savefig(os.path.join(pwd, fname), transparent=True)
    fname_pdf = "PCs of {} State {}-centroid.pdf".format(titles[j], g)
    fig_pca_per_state.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%% for 10 states, plot the distance between groups in each epoch

# First, Get the centroids of each epoch of each population in latent space.
centroids = []  # epoch 1 control, epoch 1 hp state 0, epoch 1 control epoch 1 hp state 1, ....epoch 2 control epoch 2 bd, ...
# [list of len 60 x 10] (3 epoch x 2 population x 10 cluster) x (10 mean on zdim)

centroids_subjects = []
# [list of len 1500 x 10] (2 pop x 25 sub/pop x 3 epoch x 10 cluster) x 10 zdim
latent_all = []  # epoch 1 control, epoch 1 hp, epoch 2 control, epoch 2 bd, ...
label_all = []
counter = 0
for epoch in range(1, 4):
    # for each population, get the centroid of each epoch
    for j, videos in enumerate([control_videos, BD_videos]):

        latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
        label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)

        latent_all.append(latent_vec_epoch_pop)  # (6, 225000, 10)
        label_all.append(label_epoch_pop)  # (6, 225000)

        for g in range(n_cluster):
            idx_g = np.where(label_epoch_pop == g)[0]
            latent_this_state_g = latent_vec_epoch_pop[idx_g, :]
            centroid = np.mean(latent_this_state_g, axis=0)
            centroids.append(centroid)
        counter += 1

for epoch in range(1, 4):
    # for each population, get the centroid of each epoch
    for j, videos in enumerate([control_videos, BD_videos]):

        latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
        label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)

        latent_all.append(latent_vec_epoch_pop)  # (6, 225000, 10)
        label_all.append(label_epoch_pop)  # (6, 225000)
        for sub in range(n_subject_in_population):
            latent_this_sub = latent_vec_epoch_pop[9000*sub : 9000*(sub +1)]
            label_this_sub = label_epoch_pop[9000*sub : 9000*(sub +1)]
            for g in range(n_cluster):
                idx_g = np.where(label_this_sub == g)[0]
                latent_this_state_g_this_sub = latent_this_sub[idx_g, :]
                centroid_sub = np.mean(latent_this_state_g_this_sub, axis=0).reshape(-1, 1)

                if np.isnan(centroid_sub).all():

                    centroid_sub = np.zeros((10,1))
                centroids_subjects.append(centroid_sub)


subject_end = [[0], [0]]
subject_latent_len = [[], []]

latent_all_ = np.vstack(latent_all)  # (1350000, 10) = 27000 each subject x 50 subjects
label_all_ = np.hstack(label_all)  # (1350000, 1)
centroids = np.squeeze(np.array(centroids))  # (60 x 10) #

# Next, I need to get the embedding of all latent in all three epochs,
# This is the PC embedding of latents, appended by embedding of the centroids
components_pop = pca.fit_transform(np.vstack((latent_all_, centroids)))  # (1350060, 3)

# This is the PC embedding of latents, appended by embedding of the centroids of each sub each latent
centroids_subjects = np.squeeze(np.array(centroids_subjects))  # (60 x 10) #
components_sub = pca.fit_transform(np.vstack((latent_all_, centroids_subjects)))  # (1351500, 3)
components_sub = components_sub[-1500:]

state_epoch_sub_centroids = np.zeros((3, 2, 25, 10, 3)) # 3 epoch x 2 pop x 25 sub/pop x 10 stat x 3pc
count1 = 0
for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        for sub in range(n_subject_in_population):
            for g in range(n_cluster):
                if  not np.any(centroids_subjects[count1]):
                    state_epoch_sub_centroids[epoch - 1, j, sub, g, :] = [nan, nan, nan]
                else:
                    state_epoch_sub_centroids[epoch-1, j, sub, g, :] = components_sub[count1]
                count1 += 1


labels_pop = list(label_all_)
centroids_labels = [-1] * 60 #placeholder
labels_pop += centroids_labels

colors = ['k', 'r']
# for 10 states, plot the distance between epochs, within and between groups
state_epoch_centroids = []
state_epoch_volume = []
state_volume_ = []
len_latent_epoch = [0, 5 * 30 * 60, 10 * 30 * 60, 15 * 30 * 60]
len_latent = [0, len(Latent_vectors[0])]

for g in range(10):
    epoch_centroids_per_subject = np.zeros((2, n_subject_in_population, 3)) # 2 pop x 25 subject each pop x 3 epoch
    epoch_centroids = [[], []]  # 2 x 3
    epoch_volume = [[], []]
    markers = ['o', '^', 'D']
    alphas = [0.4, 0.7, 1]
    count = 0
    idx_g = np.where(label_all_ == g)[0]
    latent_vec_all_epoch = latent_all_[idx_g, :]
    nK_g = latent_vec_all_epoch.T @ latent_vec_all_epoch
    volume_of_group = np.trace(nK_g)
    state_volume_.append(volume_of_group)

    for epoch in range(1, 4):
        fig_pca_per_state = plt.figure(figsize=(15, 10))
        ax = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
        # for each population, get the centroid of each epoch
        for j, videos in enumerate([control_videos, BD_videos]):

            latent_vec = latent_all[count]  # (1, 225000, 10)
            label_vec = label_all[count]  # (1, 225000)

            latent_vec_this_population = latent_vec
            # Then, I need to get the index of state g in label_vec
            idx = np.where(label_vec == g)[0]
            latent_vec_g = latent_vec[idx, :]
            print('latent_vec state{}-epoch{}-population{} shape: {}'.format(g, epoch, j, latent_vec_g.shape))

            # Now need to find these idx  in the (1350000, 10) array of latent_vec,
            # which is how many len_latent_epoch[225000] is away from the start
            idx_g = len_latent_each_epoch_each_population * count + idx
            count += 1

            centroid_idx = n_cluster * (j + (epoch - 1) * 2) + g
            # the last 60 elements of this list embedding of the centroids (60 x 3)
            # = (3 epoch x 2 population x 10 cluster) x pc3
            transformed_centroid = components_pop[centroid_idx - 60, :]
            epoch_centroids[j].append(transformed_centroid)
            # [centroid of HP epoch 1, ...]
            # [centroid of BD epoch 1, ...]

            principalDf_pop = pd.DataFrame(data=components_pop)
            principalDf_pop_labels = pd.DataFrame(data=labels_pop)
            finalDf = pd.concat([principalDf_pop, principalDf_pop_labels], axis=1)
            finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
            if len(latent_vec_g):
                ng_K1 = latent_vec_g.T @ latent_vec_g
                volume_of_group_epoch = np.trace(ng_K1)
                epoch_volume[j].append(volume_of_group_epoch)
                # Plot latent swarms
                ax.plot3D(components_pop[idx_g, 0], components_pop[idx_g, 1], components_pop[idx_g, 2],
                          # norm=plt.Normalize(vmin=0, vmax=9),
                          color=cmap(g * 2 + j),
                          # s=5,
                          alpha=0.9,
                          label='%d' % g, linewidth=10, zorder=-1)
            # plot centroid
            ax.scatter3D(transformed_centroid[0], transformed_centroid[1], transformed_centroid[2],
                         norm=plt.Normalize(vmin=0, vmax=9),
                         color=colors[j],
                         marker='.',
                         s=200,
                         label='{} epoch{} centroid'.format(titles[j], epoch), zorder=1)
        # plot distance between centroids between groups in same epoch
        xs = [epoch_centroids[0][epoch - 1][0], epoch_centroids[1][epoch - 1][0]]
        ys = [epoch_centroids[0][epoch - 1][1], epoch_centroids[1][epoch - 1][1]]
        zs = [epoch_centroids[0][epoch - 1][2], epoch_centroids[1][epoch - 1][2]]
        ax.plot3D(xs, ys, zs, linewidth=3, linestyle='--',
                  color='m', label='epoch{}centroid between BD-HP'.format(epoch), zorder=1)




        ax.set_title("PCs of {}-State-{}-Epoch-{} \n".format(titles[j], g, epoch))
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        # make simple, bare axis lines through space:
        # xAxisLine = ((np.min(components_pop[:, 0]), np.max(components_pop[:, 0])), (0, 0), (0, 0))
        # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
        # yAxisLine = ((0, 0), (np.min(components_pop[:, 1]), np.max(components_pop[:, 1])), (0, 0))
        # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
        # zAxisLine = ((0, 0), (0, 0), (np.min(components_pop[:, 2]), np.max(components_pop[:, 2])))
        # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')

        ax.set_xlim(-50, 50)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-40, 40)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        fig_pca_per_state.show()

        pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_centroid'.format(onedrive_path, project_name)
        Path(pwd).mkdir(parents=True, exist_ok=True)
        fname = "PCs of State {}-epoch{}-centroid.png".format(g, epoch)
        #fig_pca_per_state.savefig(os.path.join(pwd, fname), transparent=True)
        fname_pdf = "PCs of State {}-epoch{}-centroid.pdf".format(g, epoch)
        #fig_pca_per_state.savefig(os.path.join(pwd, fname_pdf), transparent=True)
    state_epoch_centroids.append(epoch_centroids)
    state_epoch_volume.append(epoch_volume)
# state_epoch_centroids = []
# state_epoch_volume = []
# for g in range(10):
#
#     epoch_centroids = [[],[]] # 2 x 3
#     epoch_volume = [[], []]
#     markers = ['o', '^', 'D']
#     alphas = [0.4, 0.7, 1]
#     for epoch in range(1, 4):
#         fig_pca_per_state_per_epoch = plt.figure(figsize=(15, 10))
#         ax = fig_pca_per_state_per_epoch.add_subplot(1, 1, 1, projection='3d')
#         # for each population, get the centroid of each epoch
#         for j, videos in enumerate([control_videos, BD_videos]):
#             latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
#             label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
#
#             i_pop = np.where(label_epoch_pop == g)
#             latent_vec_epoch_pop_this_state = latent_vec_epoch_pop[label_epoch_pop == g]
#             centroid = np.array(np.mean(latent_vec_epoch_pop, axis=0))[np.newaxis, :]
#
#             components_pop = pca.fit_transform(np.vstack((latent_vec_epoch_pop_this_state, centroid)))
#
#             transformed_centroid = components_pop[-1, :]
#             components_pop = components_pop[:-1, :]
#             epoch_centroids[j].append(transformed_centroid)
#
#             total_var_pop = pca.explained_variance_ratio_.sum() * 100
#             principalDf_pop = pd.DataFrame(data=components_pop)
#             finalDf = pd.concat([principalDf_pop, pd.DataFrame(label_epoch_pop[label_epoch_pop == g])], axis=1)
#             finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
#             if len(latent_vec_epoch_pop_this_state):
#                 K = np.cov(latent_vec_epoch_pop_this_state.T)
#                 volume_of_group = np.trace(K)
#                 epoch_volume[j].append(volume_of_group)
#                 # Plot latent swarms
#                 ax.plot3D(components_pop[:, 0], components_pop[:, 1], components_pop[:, 2],
#                           # norm=plt.Normalize(vmin=0, vmax=9),
#                            color=cmap(g * 2 + j),
#                             # s=5,
#                              alpha=1,
#                              label='%d' % g,linewidth=10, zorder=-1)
#             # plot centroid
#             ax.scatter3D(transformed_centroid[0], transformed_centroid[1], transformed_centroid[2],
#                        norm=plt.Normalize(vmin=0, vmax=9),
#                        color=colors[j],
#                        marker='.',
#                        s=200,
#                        label='{} epoch{} centroid'.format(titles[j], epoch),zorder=1)
#         # plot distance between centroids between groups in same epoch
#         xs = [epoch_centroids[0][epoch - 1][0], epoch_centroids[1][epoch - 1][0]]
#         ys = [epoch_centroids[0][epoch - 1][1], epoch_centroids[1][epoch - 1][1]]
#         zs = [epoch_centroids[0][epoch - 1][2], epoch_centroids[1][epoch - 1][2]]
#         ax.plot3D(xs, ys, zs, linewidth=3, linestyle='--',
#                 color='m', label='epoch{}centroid between BD-HP'.format(epoch),zorder=1)
#
#         ax.set_title("PCs of {}-State-{} \n".format(titles[j], g))
#         ax.set_xlabel('PC 1')
#         ax.set_ylabel('PC 2')
#         ax.set_zlabel('PC 3')
#         # make simple, bare axis lines through space:
#         # xAxisLine = ((np.min(components_pop[:, 0]), np.max(components_pop[:, 0])), (0, 0), (0, 0))
#         # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
#         # yAxisLine = ((0, 0), (np.min(components_pop[:, 1]), np.max(components_pop[:, 1])), (0, 0))
#         # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
#         # zAxisLine = ((0, 0), (0, 0), (np.min(components_pop[:, 2]), np.max(components_pop[:, 2])))
#         # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
#
#         ax.set_xlim(-50, 50)
#         ax.set_ylim(-30, 30)
#         ax.set_zlim(-40, 40)
#
#         ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         ax.tick_params(left=False, right=False, labelleft=False,
#                        labelbottom=False, bottom=False)
#
#         fig_pca_per_state_per_epoch.show()
#
#         pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_centroid'.format(onedrive_path, project_name)
#         Path(pwd).mkdir(parents=True, exist_ok=True)
#         fname = "PCs of State {}-epoch{}-centroid.png".format(g, epoch)
#         fig_pca_per_state_per_epoch.savefig(os.path.join(pwd, fname), transparent=True)
#         fname_pdf = "PCs of State {}-epoch{}-centroid.pdf".format(g, epoch)
#         fig_pca_per_state_per_epoch.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%% plot distance between centroids between groups over time
for state in range(n_cluster):
    fig, axes = plt.subplots(4, 1, figsize=(5, 15))
    this_state_centroids = state_epoch_centroids[state]
    epoch_volume = state_epoch_volume[state] # 30 x 2 x 3
    epoch_volume_list = [[],[]]
    epoch_volume_normalize_list = [[], []]
    d_zit_t_minus_BD = []
    d_zit_t_minus_HP = []
    d_zit_t_minus_BD_new = []
    d_zit_t_minus_HP_new = []
    se_d_zit_t_minus_BD = []
    se_d_zit_t_minus_HP = []
    d_zit_BD_HP_list = []
    for epoch in range(1,4):
        x = [0, 1, 2]
        d_zit_BD_HP = np.linalg.norm(this_state_centroids[0][epoch -1] - this_state_centroids[1][epoch -1])/ (state_volume[state])
        d_zit_BD_mean = np.nanmean(state_epoch_sub_centroids[epoch - 1, 1, :, state], axis=0)  # 3 epoch x 2 pop x 25 sub/pop x 10 stat x 3pc
        d_zit_HP_mean = np.nanmean(state_epoch_sub_centroids[epoch - 1, 0, :, state], axis=0)  # 3 epoch x 2 pop x 25 sub/pop x 10 stat x 3pc

        d_zit_BD_HP_test = np.linalg.norm(d_zit_BD_mean - d_zit_HP_mean)/ (state_volume[state])
        d_zit_BD_HP_se_test  = np.std((d_zit_BD_mean - d_zit_HP_mean)/state_volume[state])/np.sqrt(3)

        d_zit_BD_HP_list.append(d_zit_BD_HP)
        if epoch <= 2:
            d_zit_t_minus_HP.append( np.linalg.norm(this_state_centroids[0][epoch] - this_state_centroids[0][epoch - 1]) / state_volume[state])
            d_zit_t_minus_BD.append( np.linalg.norm(this_state_centroids[1][epoch] - this_state_centroids[1][epoch - 1]) / state_volume[state])

            d_BD = []
            d_HP = []
            for sub in range(25):
                if np.isnan(state_epoch_sub_centroids[epoch, 1, sub, state]).all() or np.isnan(state_epoch_sub_centroids[epoch-1, 1, sub, state]).all():
                    d_BD.append(nan)
                else:
                    d_BD.append(np.linalg.norm((state_epoch_sub_centroids[epoch, 1, sub, state] - state_epoch_sub_centroids[epoch - 1, 1, sub, state])/(state_volume[state])))
            for sub in range(25):
                if np.isnan(state_epoch_sub_centroids[epoch, 0, sub, state]).all() or np.isnan(state_epoch_sub_centroids[epoch-1, 0, sub, state]).all():
                    d_HP.append(nan)
                else:
                    d_HP.append(np.linalg.norm((state_epoch_sub_centroids[epoch, 0, sub,state] - state_epoch_sub_centroids[epoch - 1, 0, sub, state])/(state_volume[state])))

            #TODO:is this right??? does not equal line 1216
            d_zit_t_minus_BD_mean = np.nanmean(d_BD)# 3 epoch x 2 pop x 25 sub/pop x 10 stat x 3pc
            d_zit_t_minus_HP_mean =  np.nanmean(d_HP)


            d_zit_t_minus_BD_se = np.nanstd(d_BD)/np.sqrt(25)
            d_zit_t_minus_HP_se = np.nanstd(d_HP)/np.sqrt(25)

            d_zit_t_minus_BD_new.append(d_zit_t_minus_BD_mean)
            d_zit_t_minus_HP_new.append(d_zit_t_minus_HP_mean)

            se_d_zit_t_minus_BD.append(d_zit_t_minus_BD_se)
            se_d_zit_t_minus_HP.append(d_zit_t_minus_HP_se)


        epoch_volume_list[0].append(epoch_volume[0][epoch-1])
        epoch_volume_list[1].append(epoch_volume[1][epoch - 1])

        epoch_volume_normalize_list[0].append(epoch_volume[0][epoch-1]/state_volume_[state])
        epoch_volume_normalize_list[1].append(epoch_volume[1][epoch - 1]/state_volume_[state])

    axes[0].plot(x, d_zit_BD_HP_list, '-o', color='m', markersize=10)
    axes[0].set_xticks(x)
    axes[0].set_title("State {} distance between BD and HP centroids".format(state))
    axes[0].grid(False)
    x2 = [0, 1]

    d_zit_t_minus_BD_new = np.asarray(d_zit_t_minus_BD_new)
    se_d_zit_t_minus_BD = np.asarray(se_d_zit_t_minus_BD)
    d_zit_t_minus_HP_new  = np.asarray(d_zit_t_minus_HP_new)
    se_d_zit_t_minus_HP = np.asarray(se_d_zit_t_minus_HP)
    axes[1].plot(x2, d_zit_t_minus_BD_new, '--o', label='BD', color=b_o_colors[1], markersize=10)
    axes[1].fill_between(x2, d_zit_t_minus_BD_new - se_d_zit_t_minus_BD, d_zit_t_minus_BD_new + se_d_zit_t_minus_BD,
                      alpha=0.2, facecolor=b_o_colors[1])

    axes[1].plot(x2, d_zit_t_minus_HP_new, '-o', label='HP', color=b_o_colors[0], markersize=10)
    axes[1].fill_between(x2, d_zit_t_minus_HP_new - se_d_zit_t_minus_HP, d_zit_t_minus_HP_new + se_d_zit_t_minus_HP,
                      alpha=0.2, facecolor=b_o_colors[0])

    axes[1].set_xticks(x2)
    axes[1].grid(False)
    axes[1].set_title("distance between centroids within population")

    axes[2].plot(x, epoch_volume_list[0], '--o', label='BD', color=b_o_colors[1], markersize=10)
    axes[2].plot(x, epoch_volume_list[1], '-o', label='HP', color=b_o_colors[0], markersize=10)
    axes[2].set_xticks(x)
    axes[2].set_title("volume of BD and HP")
    axes[2].grid(False)

    axes[3].plot(x, epoch_volume_normalize_list[0], '--o', label='BD', color=b_o_colors[1], markersize=10)
    axes[3].plot(x, epoch_volume_normalize_list[1], '-o', label='HP', color=b_o_colors[0], markersize=10)
    axes[3].set_xticks(x)
    axes[3].set_title("relative volume portion/volume(state)")
    axes[3].grid(False)

    fig.show()




    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_centroid'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "State {}-centroid-distance.png".format(state)
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fname_pdf = "State {}-centroid-distance.pdf".format(state)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)


#%%  PCs of {}-State-{}-Epoch-{}
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
pca = PCA(n_components=3)
K_var = [np.zeros((10, 3)), np.zeros((10, 3))]
K_var_all_subjects = [np.zeros((n_subject_in_population,10, 3)), np.zeros((n_subject_in_population, 10, 3))]
for j, videos in enumerate([control_videos, BD_videos]):
    alphas = [0.4, 0.7, 1]
    for epoch in range(1, 4):
        latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for sub in range(n_subject_in_population):
            sub_name = videos[sub]
            latent_vec_epoch_sub = eval('Epoch{}_latent_vector'.format(epoch))[j][sub]
            label_epoch_sub = eval('Epoch{}_labels'.format(epoch))[j][sub]
            for g in np.unique(label):
                i_sub = np.where(label_epoch_sub == g)
                i_pop = np.where(label_epoch_pop == g)
                latent_vec_epoch_sub_this_state = latent_vec_epoch_sub[label_epoch_sub == g]
                latent_vec_epoch_pop_this_state = latent_vec_epoch_pop[label_epoch_pop == g]

                if len(i_sub[0]) and latent_vec_epoch_sub_this_state.shape[0] > 2:
                    components = pca.fit_transform(latent_vec_epoch_sub_this_state)
                    total_var = pca.explained_variance_ratio_.sum() * 100
                    principalDf = pd.DataFrame(data=components)
                    finalDf = pd.concat([principalDf, pd.DataFrame(label_epoch_sub[label_epoch_sub == g])], axis=1)
                    finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
                    K_sub = np.cov(latent_vec_epoch_sub_this_state.T)
                    volume_of_epoch_sub_this_state = np.trace(K_sub)
                    # plot this usbject, this epoch, this state

                    fig_pca_per_sub_per_state = plt.figure(figsize=(10, 10))
                    ax2 = fig_pca_per_sub_per_state.add_subplot(1, 1, 1, projection='3d')
                    ax2.plot3D(components[:, 0], components[:, 1], components[:, 2],
                               linewidth=10,
                                #norm=plt.Normalize(vmin=0, vmax=9),s=5,
                                color=cmap(g * 2 + j),  alpha=alphas[epoch-1], label='%d' % g)
                    ax2.set_title(
                        "PCs of {}-{}-State-{}-epoch-{}\n Exp_Var:{:.2f}".format(titles[j], sub_name, g, epoch,
                                                                                 total_var))
                    ax2.set_xlabel('PC 1')
                    ax2.set_ylabel('PC 2')
                    ax2.set_zlabel('PC 3')
                    # make simple, bare axis lines through space:
                    xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
                    ax2.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
                    yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
                    ax2.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
                    zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
                    ax2.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
                    ax2.set_xlim(-50, 50)
                    ax2.set_ylim(-30, 30)
                    ax2.set_zlim(-40, 40)
                    fig_pca_per_sub_per_state.show()

                    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch\subject_each_state'.format(onedrive_path,
                                                                                                       project_name)
                    Path(pwd).mkdir(parents=True, exist_ok=True)
                    fname = "PCs of {}-{} State {}-epoch{}.png".format(titles[j], sub_name, g, epoch)
                    fname_pdf = "PCs of {}-{} State {}-epoch{}.pdf".format(titles[j], sub_name, g, epoch)
                    fig_pca_per_sub_per_state.savefig(os.path.join(pwd, fname))
                    fig_pca_per_sub_per_state.savefig(os.path.join(pwd, fname_pdf))
                else:
                    volume_of_epoch_sub_this_state = 0
                    print("{}-{}-epoch {} has no latent for state {}".format(titles[j], sub_name, epoch, g))
                K_temp = K_var_all_subjects[j]
                K_temp[sub][g][epoch - 1] = volume_of_epoch_sub_this_state
                K_var_all_subjects[j] = K_temp



                # Plot this population, this epoch, this state
                components_pop = pca.fit_transform(latent_vec_epoch_pop_this_state)



                components_pop = pca.fit_transform(np.vstack((latent_vec_epoch_pop_this_state, centroid)))

                total_var_pop = pca.explained_variance_ratio_.sum() * 100
                principalDf_pop = pd.DataFrame(data=components_pop)
                finalDf = pd.concat([principalDf_pop, pd.DataFrame(label_epoch_pop[label_epoch_pop == g])], axis=1)
                finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
                if len(latent_vec_epoch_pop_this_state):
                    K = np.cov(latent_vec_epoch_pop_this_state.T)
                    volume_of_group = np.trace(K)
                    fig_pca_per_state = plt.figure(figsize=(10, 10))
                    ax = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')

                    # ax.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
                    #            color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
                    ax.plot3D(components_pop[:, 0], components_pop[:, 1], components_pop[:, 2],
                               linewidth=10,
                               # norm=plt.Normalize(vmin=0, vmax=9),s=5,
                               color=cmap(g * 2 + j), alpha=alphas[epoch - 1], label='%d' % g)



                    ax.set_title(
                        "PCs of {}-State-{}-Epoch-{} \n volume:{:.2f}".format(titles[j], g, epoch, volume_of_group))
                    ax.tick_params(left=False, right=False, labelleft=False,
                                    labelbottom=False, bottom=False)
                    ax.set_xlabel('PC 1')
                    ax.set_ylabel('PC 2')
                    ax.set_zlabel('PC 3')
                    # make simple, bare axis lines through space:
                    # xAxisLine = ((np.min(components_pop[:, 0]), np.max(components_pop[:, 0])), (0, 0), (0, 0))
                    # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
                    # yAxisLine = ((0, 0), (np.min(components_pop[:, 1]), np.max(components_pop[:, 1])), (0, 0))
                    # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
                    # zAxisLine = ((0, 0), (0, 0), (np.min(components_pop[:, 2]), np.max(components_pop[:, 2])))
                    # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
                    ax.set_xlim(-50, 50)
                    ax.set_ylim(-30, 30)
                    ax.set_zlim(-40, 40)

                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    fig_pca_per_state.show()

                    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch'.format(onedrive_path, project_name)
                    Path(pwd).mkdir(parents=True, exist_ok=True)
                    fname = "PCs of {} State {} Epoch {}.png".format(titles[j], g, epoch)
                    fig_pca_per_state.savefig(os.path.join(pwd, fname), transparent=True)
                    fname_pdf = "PCs of {} State {} Epoch {}.pdf".format(titles[j], g, epoch)
                    fig_pca_per_state.savefig(os.path.join(pwd, fname_pdf), transparent=True)
                else:
                    volume_of_group = 0
                    print("{}-{} has no latent for state {}".format(titles[j], epoch, g))
                K_var[j][g][epoch - 1] = volume_of_group




#%%
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
pca = PCA(n_components=3)
K_var = [np.zeros((10, 3)), np.zeros((10, 3))]
K_var_all_subjects = [np.zeros((n_subject_in_population,10, 3)), np.zeros((n_subject_in_population, 10, 3))]
for j, videos in enumerate([control_videos, BD_videos]):
    for epoch in range(1, 4):
        # latent vector of the epoch
        latent_vec = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0) # [25 x 9000 ]x 10 = 225000 x 10
        latent_vec_trim = latent_vec #  2x 25 x 9000 x 10
        label = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        label_trim = label
        components = pca.fit_transform(latent_vec)
        total_var = pca.explained_variance_ratio_.sum() * 100
        principalDf = pd.DataFrame(data=components)
        finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
        finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']

        latent_vec_pop = latent_vec_trim[0: Latent_len_epoch[epoch - 1][j][:]]
        latent_vec_pop_trim = latent_vec_trim[Latent_len_epoch[epoch - 1][j][:]:]

        label_pop = label_trim[0: Latent_len_epoch[epoch - 1][j][sub]]
        label_pop_trim = label_trim[Latent_len_epoch[epoch - 1][j][sub]:]
        for sub in range(n_subject_in_population):
            latent_vec_sub = latent_vec_trim[0: Latent_len_epoch[epoch-1][j][sub]]
            latent_vec_trim = latent_vec_trim[Latent_len_epoch[epoch-1][j][sub]:]

            label_sub = label_trim[0: Latent_len_epoch[epoch-1][j][sub]]
            label_trim = label_trim[Latent_len_epoch[epoch-1][j][sub]:]
            sub_name = videos[sub]
            for g in np.unique(label):
                # state-subject-plot

                i_sub = np.where(label_sub == g)
                i_pop = np.where(label_pop == g)
                latent_pop_g = latent_vec_pop[i_pop]
                components_pop = pca.fit_transform(latent_pop_g)
                total_var_pop = pca.explained_variance_ratio_.sum() * 100


                if len(i_sub[0]) and latent_vec_sub[i_sub].shape[0] > 2:
                    # the latent for this subject, this epoch, this state
                    latent_sub_g = latent_vec_sub[i_sub]
                    components = pca.fit_transform(latent_sub_g)
                    total_var = pca.explained_variance_ratio_.sum() * 100
                    principalDf = pd.DataFrame(data=components)
                    finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
                    finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
                    K_sub = np.cov(latent_sub_g.T)
                    volume_of_group_sub = np.trace(K_sub)

                else:
                    volume_of_group_sub = 0

                K_temp = K_var_all_subjects[j]
                K_temp[sub][g][epoch - 1] = volume_of_group_sub
                K_var_all_subjects[j] = K_temp
                fig_pca_per_sub_per_state = plt.figure(figsize=(10, 10))
                ax2 = fig_pca_per_sub_per_state.add_subplot(1, 1, 1, projection='3d')

                # ax.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
                #            color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
                ax2.scatter(components[:, 0], components[:, 1], components[:, 2],
                            norm=plt.Normalize(vmin=0, vmax=9),
                            color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)

                ax2.set_title("PCs of {}-{}-State-{}-epoch-{}\n Exp_Var:{:.2f}".format(titles[j], sub_name, g, epoch, total_var))
                ax2.set_xlabel('PC 1')
                ax2.set_ylabel('PC 2')
                ax2.set_zlabel('PC 3')

                # make simple, bare axis lines through space:
                xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
                ax2.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
                yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
                ax2.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
                zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
                ax2.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
                ax2.set_xlim(-40, 40)
                ax2.set_ylim(-30, 30)
                ax2.set_zlim(-40, 40)
                fig_pca_per_sub_per_state.show()
                pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch\subject_each_state'.format(onedrive_path, project_name)
                Path(pwd).mkdir(parents=True, exist_ok=True)
                fname = "PCs of {}-{} State {}-epoch{}.png".format(titles[j], sub_name, g, epoch)
                fname_pdf = "PCs of {}-{} State {}-epoch{}.pdf".format(titles[j], sub_name, g, epoch)
                fig_pca_per_sub_per_state.savefig(os.path.join(pwd, fname))
                fig_pca_per_sub_per_state.savefig(os.path.join(pwd, fname_pdf))

                # state-population plot (idea: state-subject overlap)
                i = np.where(label == g)
                latent_vec_g = latent_vec[i]
                if len(latent_vec_g):
                    K = np.cov(latent_vec_g.T)
                    volume_of_group = np.trace(K)
                else:
                    volume_of_group = 0
                K_var[j][g][epoch-1] = volume_of_group
                fig_pca_per_state = plt.figure(figsize=(10, 10))
                ax = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')

                # ax.scatter(components[i, 0], components[i, 1], components[i, 2],norm=plt.Normalize(vmin=0, vmax=9),s=3, color=cmap(g * 2 + j), s=3, alpha=0.1, label='%d' % g)
                ax.scatter(components[i, 0], components[i, 1], components[i, 2],
                           norm=plt.Normalize(vmin=0, vmax=9),
                           s=3,
                           color=cmap(g * 2 + j),
                           alpha=0.1,
                           label='%d' % g)

                ax.set_title("PCs of {}-State-{}-Epoch-{} \n volume:{:.2f}".format(titles[j], g,epoch, volume_of_group))
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                ax.set_zlabel('PC 3')
                # make simple, bare axis lines through space:
                xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
                ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
                yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
                ax2.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
                zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
                ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
                ax.set_xlim(-50, 50)
                ax.set_ylim(-30, 30)
                ax.set_zlim(-40, 40)
                print("I am here")
                pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch'.format(onedrive_path, project_name)
                Path(pwd).mkdir(parents=True, exist_ok=True)
                fname = "PCs of {} State {} Epoch {}.png".format(titles[j], g, epoch)
                fig_pca_per_state.savefig(os.path.join(pwd, fname), transparent=True)
                fname_pdf = "PCs of {} State {} Epoch {}.pdf".format(titles[j], g, epoch)
                fig_pca_per_state.savefig(os.path.join(pwd, fname_pdf), transparent=True)

#%%
for g in range(n_cluster):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(3)
    ax.plot(x, K_var[0][g],'-o')
    ax.plot(x, K_var[1][g],'-o')
    ax.set_title('Volume of state {}'.format(g))
    ax.set_xticks(x, ['Epoch 1', 'Epoch 2', 'Epoch 3'])
    ax.set_xlabel('Epoch (5min)')
    ax.set_ylabel('volume of state')
    ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\volume'.format(onedrive_path, project_name)
    fname = "Volume of state {}.png".format(g)
    fig.savefig(os.path.join(pwd, fname))
#%%
for epoch in range(3):
    for j, videos in enumerate([control_videos, BD_videos]):
        color = 'C{}'.format(j)

        for sub in range(12):
            sub_name = videos[sub]
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            x = np.arange(10)
            Ktemp = K_var_all_subjects[j]
            ax.scatter(x,Ktemp[sub,:,epoch], c=color)
            ax.set_title('{} Volume of state epoch {}'.format(sub_name, epoch+1))
            ax.set_xticks(x)
            ax.set_xlabel('State')
            ax.set_ylabel('volume of state')
            ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
            fig.show()
            pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\volume'.format(onedrive_path, project_name)
            fname = "{}_{}_epoch{}_volume.png".format(sub_name, n_cluster, epoch+1)
            fig.savefig(os.path.join(pwd, fname))
            fname_pdf = "{}_{}_epoch{}_volume.pdf".format(sub_name, n_cluster, epoch+1)
            fig.savefig(os.path.join(pwd, fname_pdf))
#%%
for epoch in range(3):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    x = np.arange(10)
    Ktemp = K_var_all_subjects[0]
    Ktemp1 = K_var_all_subjects[1]
    for sub in range(12):
        ax.scatter(x, Ktemp[sub, :, epoch], c='C0', alpha=0.5)
        ax.scatter(x+0.2, Ktemp1[sub, :, epoch], c='C1', alpha=0.5)
        x = np.arange(10)

    ax.plot(x, np.median(Ktemp,axis=[0,2]), '-^', color='C0')
    ax.plot(x, np.median(Ktemp1,axis=[0,2]),'-^', color='C1')
    ax.set_title('Volume of state epoch {}'.format(epoch))
    ax.set_xticks(x)
    ax.set_xlabel('State')
    ax.set_ylabel('volume of state')
    ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual'.format(onedrive_path, project_name)
    fname = "{}_volume_epoch_{}.png".format(n_cluster, epoch)
    fig.savefig(os.path.join(pwd, fname))
