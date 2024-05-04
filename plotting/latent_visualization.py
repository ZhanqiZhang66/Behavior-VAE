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
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import distance
from dtaidistance import dtw
from plotting.get_paths import get_my_path
from dtaidistance import dtw_ndim
matplotlib.use('Qt5Agg')
#%%
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
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(project_path,"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
zdim = 10
model_name = 'VAME'
five_min_frame_no = int(5 * 60 * 30)
data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=
                                                                       r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
score_bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
n_subject_in_population = len(control_videos)
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
pca = PCA(n_components=3)
#%%
'''
Define Variables of interests
'''
titles = ["CP", "BD"]
N = [0, 0]


Motif_usages = [[], []]
Motif_usage_pct = [[], []]
motif_usage_cat = [[], []]
Latent_vectors = [[], []]
Latent_centroids = [[], []]  # this is the mean of the latent, on d-dimension
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
'''
Step 1: Mean-centered the population-latent vectors for computing
'''

# Get the latent vector of each video
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Loading {} data...".format(v))

        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))

        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # T x d = {10, 30}
        latent_vector = latent_vector[: five_min_frame_no*3]
        label = label[: five_min_frame_no*3]
        if i == 0:
            l = label
            latent = latent_vector
        else:
            latent = np.concatenate([latent, latent_vector])
            l = np.concatenate([l,label])
    Latent_len[j].append(len(latent_vector))
    Latent_vectors[j] = latent
    Labels[j] = l

# compute the mean of latent population
for j, videos in enumerate([control_videos, BD_videos]):
    population_latent_vector = Latent_vectors[j]
    population_latent_vector_centroid = np.mean(population_latent_vector, axis=0)
    Latent_centroids[j] = population_latent_vector_centroid

#%%
'''
Step 2: compute overall latent motif-volume, and epoch latent motif-volume
'''
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Mean-Centering {} data...".format(v))
        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        motif_usage = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # T x d = {10, 30}
        Latent_len[j].append(len(latent_vector))

        # variance of each person
        K = np.cov(latent_vector.T)  # covariance matrix of the 10-D latent vector
        volume = np.trace(K)
        volume_per_person[j].append(volume)

        door_close_time = start_frame[v]
        start_time = door_close_time

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

        # mean-centered epoch latent vector
        epoch_1_latent_vector = latent_vector[:five_min_frame_no + offset] - Latent_centroids[j]
        epoch_2_latent_vector = latent_vector[five_min_frame_no + offset: five_min_frame_no * 2 + offset] - Latent_centroids[j]
        epoch_3_latent_vector = latent_vector[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]- Latent_centroids[j]


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
    #
    #     if i == 0:
    #         l = label
    #         latent = latent_vector
    #     else:
    #         latent = np.concatenate([latent, latent_vector])
    #         l = np.concatenate([l,label])
    #
    #     motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
    #     num_points = latent_vector.shape[0]
    #     n += num_points
    #
    # latent_variance_per_state = []
    # stats_per_latent_per_state = np.empty((n_cluster, np.shape(latent_vector)[1], 2))  # 10 x 10
    # for state_i in range(n_cluster):
    #     state_i_idx = label == state_i
    #     latent_state_i = latent_vector[state_i_idx, :]  # time of (state i) x 10
    #     K_per_state = np.cov(latent_state_i.T)
    #     latent_variance_per_state.append([np.mean(latent_state_i), np.trace(K_per_state)])
    #     for latent_d in range(np.shape(latent_vector)[1]):
    #         latent_d_state_i = latent_state_i[:, latent_d]
    #         stats_per_latent_per_state[state_i, latent_d, 0] = np.nanmean(latent_d_state_i)
    #         stats_per_latent_per_state[state_i, latent_d, 1] = np.nanvar(latent_d_state_i)
    #
    # Stats_per_latent_per_state[j].append(stats_per_latent_per_state)
    # Latent_vectors_per_state[j].append(latent_variance_per_state)
    #
    # N[j] = n
    # Latent_vectors[j] = latent
    # Labels[j] = l
    #
#%%
jack_path = r'C:\Users\zhanq\OneDrive - UC San Diego\Behavior_VAE_data\jack_temp'

np.save(jack_path + '\Epoch1_labels', np.array(Epoch1_labels, dtype=object))
np.save(jack_path + '\Epoch1_latent_vector', np.array(Epoch1_latent_vector, dtype=object))
np.save(jack_path + '\Epoch2_labels', np.array(Epoch2_labels, dtype=object))
np.save(jack_path + '\Epoch2_latent_vector', np.array(Epoch2_latent_vector, dtype=object))
np.save(jack_path + '\Epoch3_labels', np.array(Epoch3_labels, dtype=object))
np.save(jack_path + '\Epoch3_latent_vector', np.array(Epoch3_latent_vector, dtype=object))

#%% #TODO: see if we can decode position latent
#%%
'''
Population-wise plot
'''
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


        fig = plt.figure(figsize=(5, 10))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        # plot trajectory on latent

        time = np.arange(0, len(components))
        c = np.linspace(0, 1, len(components))
        ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=c, cmap=plt.get_cmap('plasma'),
                   label='trajectory', s=10, alpha=0.3)
        # make simple, bare axis lines through space:
        # xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
        # ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
        # yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
        # ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
        # zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
        # ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)
        # ax.set_xlabel('PC 1 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[0]))
        # ax.set_ylabel('PC 2 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[1]))
        # ax.set_zlabel('PC 3 Exp_Var:{:.2f}'.format(pca.explained_variance_ratio_[2]))

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
                       color=cmap(g * 2 + 0),s=10, label='%d' % g, alpha=0.3)
        leg =  ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # make simple, bare axis lines through space:
        # xAxisLine = ((np.min(components[:, 0]), np.max(components[:, 0])), (0, 0), (0, 0))
        # ax1.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'k--')
        # yAxisLine = ((0, 0), (np.min(components[:, 1]), np.max(components[:, 1])), (0, 0))
        # ax1.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'k--')
        # zAxisLine = ((0, 0), (0, 0), (np.min(components[:, 2]), np.max(components[:, 2])))
        # ax1.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'k--')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        ax1.set_title("PCs of {}-{}\n Exp_Var:{:.2f}".format(titles[j], v, total_var))


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
fig_pca = plt.figure(figsize=(5,10))
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
                   s=10, alpha=0.3, label='%d' % g)
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
#%% Plot volume of latent variable each epoch

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
#%%
'''
Epoch-wise plot
'''

#%% Helper function
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
cmap = plt.get_cmap('tab20')
# First, Get the centroids of each epoch of each population
centroids = []
# epoch 1 hc state 0,    idx = 0
# epoch 1 hc state 1,    idx = 1
# epoch 1 hc state 2,    idx = 2
# ....
# epoch 1 bd state 0, ...idx = 10
# epoch 1 bd state 1, ...idx = 11
# epoch x pop j state g, ...idx = n_cluster * (j + (epoch-1)*2) + g
centroids_se = []
latent_all = [] # epoch 1 control, epoch 1 hp, epoch 2 control, epoch 2 bd, ...
label_all = []
counter = 0
for epoch in range(1, 4):
    # for each population, get the centroid of each epoch
    for j, videos in enumerate([control_videos, BD_videos]):
        # (9000x25) x d = 225000 x d
        latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
        # 225000 x 1
        label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)

        latent_all.append(latent_vec_epoch_pop) # (6, 225000, 10)
        label_all.append(label_epoch_pop) # (6, 225000)

        for g in range(n_cluster):
            idx_g = np.where(label_epoch_pop == g)[0]
            centroid = np.nanmean(latent_vec_epoch_pop[idx_g, :], axis=0)
            centroids.append(centroid)
        counter += 1

subject_end = [[0],[0]]
subject_latent_len = [[],[]]

latent_all_ = np.vstack(latent_all) # (1350000, 10)
label_all_ = np.hstack(label_all) # (1350000, 1)
centroids = np.squeeze(np.array(centroids)) # (60 x 10)  (3 epoch x 2 pop x 10 motif) x d

# Next, I need to get the embedding of all latent in all three epochs

components = pca.fit_transform(latent_all_)
components_pop = pca.transform(centroids)# (60, 3)
labels_pop = list(label_all_)



colors = b_o_colors
#%% for each of the 10 motifs/states, we plot the centroids of all epochs, within and between populations

state_epoch_centroids = []
state_epoch_volume = []
len_latent_epoch = [0, 5*30*60, 10*30*60, 15*30*60]
len_latent = [0, len(Latent_vectors[0])]
count_centroid = 0
# for each motif
for g in range(10):
    fig_pca_per_state = plt.figure(figsize=(6, 4))
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
            # this is the latent vector of this population, in this motif, of this epoch
            idx = np.where(label_vec == g)[0]
            latent_vec_g = latent_vec[idx, :]
            print('latent_vec_g shape: {}'.format(latent_vec_g.shape))

            transformed_centroid = components_pop[count, :]
            epoch_centroids[j].append(transformed_centroid)
            count += 1
            # centroid of motif 1 epoch 1 HC,
            # centroid of motif 1 epoch 1 BD , ...

            principalDf_pop = pd.DataFrame(data=components)
            principalDf_pop_labels = pd.DataFrame(data=labels_pop)
            finalDf = pd.concat([principalDf_pop, principalDf_pop_labels], axis=1)
            finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
            if len(latent_vec_g):
                # motif-volume of this population, this epoch
                KK = np.cov(latent_vec_g.T)
                volume_of_group = np.trace(KK)
                epoch_volume[j].append(volume_of_group)
                # Plot latent swarms
                ax.plot3D(components[idx_g, 0], components[idx_g, 1], components[idx_g, 2],
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
        # plot distance between centroids between groups in same epoch in PC space
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
                lw=3, arrowstyle="-|>", color='k')
    ax.add_artist(a)
    # ax.plot3D(xs, ys, zs, linestyle='-', linewidth=5, color='k',
    #         label='HP centroid between epoch 1 and 2',zorder=1)
    xs = [epoch_centroids[0][1][0], epoch_centroids[0][2][0]]
    ys = [epoch_centroids[0][1][1], epoch_centroids[0][2][1]]
    zs = [epoch_centroids[0][1][2], epoch_centroids[0][2][2]]
    a = Arrow3D(xs, ys,
                zs, mutation_scale=20,
                lw=3, arrowstyle="-|>", color='k')
    ax.add_artist(a)

    xs = [epoch_centroids[1][0][0], epoch_centroids[1][1][0]]
    ys = [epoch_centroids[1][0][1], epoch_centroids[1][1][1]]
    zs = [epoch_centroids[1][0][2], epoch_centroids[1][1][2]]
    a = Arrow3D(xs, ys,
                zs, mutation_scale=20,
                lw=3, arrowstyle="-|>", color='r')
    ax.add_artist(a)
    xs = [epoch_centroids[1][1][0], epoch_centroids[1][2][0]]
    ys = [epoch_centroids[1][1][1], epoch_centroids[1][2][1]]
    zs = [epoch_centroids[1][1][2], epoch_centroids[1][2][2]]
    a = Arrow3D(xs, ys,
                zs, mutation_scale=20,
                lw=3, arrowstyle="-|>", color='r')
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
#%% We also need the centroid of each subject, for error bars
'''
latent per subject
'''
# First, Get the centroids of each epoch of each population in latent space.
centroids = []
# [list of len 60 x 10] (10 motifs x 3 epoch x 2 population ) x (10 mean on zdim)

centroids_subjects = []
# [list of len 1500 x 10] (3 epoch x 2 pop x 25 sub/pop x 10 motifs) x 10 zdim
volume_subjects = []
# [list of 1500 x 10] (3 epoch x 2 pop x 25 sub/pop x 10 motifs) x 1
latent_all = []  # epoch 1 control, epoch 1 bd, epoch 2 control, epoch 2 bd, ...
label_all = []

centroids_reshape = np.zeros((3, 2, n_cluster, zdim))
for g in range(n_cluster):
    for epoch in range(1, 4):
        # for each population, get the centroid of each epoch
        for j, videos in enumerate([control_videos, BD_videos]):
            latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
            len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
            label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
            latent_all.append(latent_vec_epoch_pop)  # (6, 225000, 10)
            label_all.append(label_epoch_pop)  # (6, 225000)
            idx_g = np.where(label_epoch_pop == g)[0]
            latent_this_state_g = latent_vec_epoch_pop[idx_g, :]
            if len(latent_this_state_g):
                centroid = np.nanmean(latent_this_state_g, axis=0)
            else:
                centroid = np.full([zdim, ], np.nan)
            centroids.append(centroid)
            centroids_reshape[epoch - 1, j, g, :] = centroid

# Then, Get the centroids of each epoch of each subject in latent space.
for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
        label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for sub in range(n_subject_in_population):
            latent_this_sub = latent_vec_epoch_pop[9000*sub: 9000*(sub +1)] #9000 x 10
            label_this_sub = label_epoch_pop[9000*sub : 9000*(sub +1)]
            for g in range(n_cluster):
                idx_g = np.where(label_this_sub == g)[0]
                latent_this_state_g_this_sub = latent_this_sub[idx_g, :]
                if len(latent_this_state_g_this_sub):
                    latent_volume_this_state_g_this_sub = np.trace(np.cov(latent_this_state_g_this_sub.T))
                    centroid_sub = np.mean(latent_this_state_g_this_sub, axis=0)
                else:
                    print(f"a empty latent volume for {videos[sub]} in motif{g} epoch{epoch}")
                    latent_volume_this_state_g_this_sub = 0
                    centroid_sub = np.full([zdim,], 0)

                centroids_subjects.append(centroid_sub)
                volume_subjects.append(latent_volume_this_state_g_this_sub)



#%% Here, we are getting the PC space coordinates centroid of each subject's latent vector, per motif, per epoch
'''
latent per subject
'''
subject_end = [[0], [0]]
subject_latent_len = [[], []]

latent_all_ = np.vstack(latent_all)  # (1350000, 10) = 27000 each subject x 50 subjects x zdim
label_all_ = np.hstack(label_all)  # (1350000, 1)
# centroids of each epoch of each population in latent space.
centroids = np.squeeze(np.array(centroids))  # (60 x 10) (3 epoch x 2 pop x 10 motif) x zdim

pca = PCA(n_components=3)
# Next, I need to get the embedding of all latent in all three epochs,
# This is the PC embedding of latents, appended by embedding of the centroids
components = pca.fit_transform(latent_all_)
components_centroids = pca.transform(centroids)# (60, 3)


# This is the PC embedding of latents, appended by embedding of the centroids of each sub each latent
centroids_subjects = np.squeeze(np.array(centroids_subjects))  # (1500 x 10) (3 epoch x 2 pop x 25 sub/pop x 10 motif) x 10
components_sub = pca.transform(centroids_subjects) # (1500, 3)

# reorganizing the datastructure
state_epoch_sub_centroids_PC = np.zeros((3, 2, n_subject_in_population, n_cluster, 3)) # 3 epoch x 2 pop x 25 sub/pop x 10 motif x 3pc
state_epoch_sub_centroids = np.zeros((3, 2, n_subject_in_population, n_cluster, zdim)) # 3 epoch x 2 pop x 25 sub/pop x 10 motif x zdim
state_epoch_sub_volume = np.zeros((3, 2, n_subject_in_population, n_cluster))
count1 = 0
count = 0

for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):

        for sub in range(n_subject_in_population):
            for g in range(n_cluster):
                if not np.any(centroids_subjects[count1]):
                    print(f"latent vector of {titles[j]} motif {g} epoch{epoch} is nan")
                    state_epoch_sub_centroids_PC[epoch - 1, j, sub, g, :] = [nan, nan, nan]
                    state_epoch_sub_centroids[epoch - 1, j, sub, g, :] = [np.nan] * 10
                    state_epoch_sub_volume[epoch - 1, j, sub, g] = 0
                else:
                    state_epoch_sub_centroids_PC[epoch-1, j, sub, g, :] = components_sub[count1]
                    state_epoch_sub_centroids[epoch-1, j, sub, g, :] = centroids_subjects[count1]
                    state_epoch_sub_volume[epoch - 1, j, sub, g] = volume_subjects[count1]
                count1 += 1
            this_population_this_epoch_centroid_2 = np.nanmean(state_epoch_sub_centroids[epoch - 1, j, :, g,:], axis=0)





#%%
'''
DTW distance between trajectories of each subject
'''
import itertools
# Initialize an empty DataFrame
df = pd.DataFrame(columns=['Subject', 'is_BD', 'Epoch', 'Motif', 'Latent_Vector'])

for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        latent_vec_this_epoch_this_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        label_this_epoch_this_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for g in range(n_cluster):
            for sub_i in range(n_subject_in_population):
                latent_vec_this_epoch_this_pop_this_person = latent_vec_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                label_this_epoch_this_pop_this_person = label_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                idx_g = np.where(label_this_epoch_this_pop_this_person == g)[0]
                latent_vec_this_epoch_this_pop_this_person_this_motif = latent_vec_this_epoch_this_pop_this_person[idx_g]
                # Create a dictionary to store the data
                data = {
                    'Subject': sub_i,
                    'is_BD': j,
                    'Epoch': epoch,
                    'Motif': g,
                    'Latent_Vector': latent_vec_this_epoch_this_pop_this_person_this_motif
                }
                # Append the data to the DataFrame
                df = df.append(data, ignore_index=True)
#%% Stop! A DTW computation that needs to run in parallel (to speed up)

# Initialize a DataFrame to store the distances
distances_BD_HC = []
distances_HC_HC = []
distances_BD_BD = []
for epoch in range(1, 4):
    for motif in range(n_cluster):
        # Select the rows corresponding to J == 0 and J == 1 for the current epoch and motif
        df_epoch_motif_HC = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 0)]
        df_epoch_motif_BD = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 1)]

        # Iterate over each pair of latent vectors and compute the distance
        print(f"motif{motif} epoch {epoch}")
        for i in range(len(df_epoch_motif_HC)):
            for j in range(len(df_epoch_motif_BD)):
                latent_vector_HC = df_epoch_motif_HC.iloc[i]['Latent_Vector']
                latent_vector_BD = df_epoch_motif_BD.iloc[j]['Latent_Vector']

                # Compute distance between the latent vectors
                BD_HC_distance = dtw_ndim.distance(latent_vector_BD, latent_vector_HC)

                # Add the distance to the distances DataFrame
                distances_BD_HC.append({'Epoch': epoch, 'Motif': motif, 'Distance': BD_HC_distance}
                                                   )
                print(f"  person{i} vs person {j}")
        # Iterate over each pair of latent vectors and compute the distance
        for i, j in itertools.combinations(range(len(df_epoch_motif_BD)), 2):
            latent_vector_HC = df_epoch_motif_HC.iloc[i]['Latent_Vector']
            latent_vector_HC2 = df_epoch_motif_HC.iloc[j]['Latent_Vector']

            # Compute distance between the latent vectors
            HC_HC_distance = dtw_ndim.distance(latent_vector_HC, latent_vector_HC2)

            # Add the distance to the distances DataFrame
            distances_HC_HC.append({'Epoch': epoch, 'Motif': motif, 'Distance': HC_HC_distance}
                                               )

        # Iterate over each pair of latent vectors and compute the distance
        for i, j in itertools.combinations(range(len(df_epoch_motif_BD)), 2):
            latent_vector_BD = df_epoch_motif_BD.iloc[i]['Latent_Vector']
            latent_vector_BD2 = df_epoch_motif_BD.iloc[j]['Latent_Vector']

            # Compute distance between the latent vectors
            BD_BD_distance = dtw_ndim.distance(latent_vector_BD, latent_vector_BD2)

            # Add the distance to the distances DataFrame
            distances_BD_BD.append({'Epoch': epoch, 'Motif': motif, 'Distance': BD_BD_distance}
                                               )
distances_BD_BD_df = pd.DataFrame(distances_BD_BD)
distances_BD_BD_df.to_csv('distances_BD_BD_df.csv', index=False)
distances_HC_HC_df = pd.DataFrame(distances_HC_HC)
distances_HC_HC_df.to_csv('distances_HC_HC_df.csv', index=False)

#%%
'''
add missing diagonal distances
'''
distances_BD_BD_df = pd.read_csv(f'{github_path}\Behavior-VAE\data\distances_BD_BD_df.csv')
distances_HC_HC_df = pd.read_csv(f'{github_path}\Behavior-VAE\data\distances_HC_HC_df.csv')
distances_HC_BD_df = pd.read_csv(f'{github_path}\Behavior-VAE\data\distances_HC_BD_df.csv')
distances_BD_HC_diagnal = []
for epoch in range(1, 4):
    for motif in range(n_cluster):
        # Select the rows corresponding to J == 0 and J == 1 for the current epoch and motif
        df_epoch_motif_HC = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 0)]
        df_epoch_motif_BD = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 1)]

        latent_vector_HC = df_epoch_motif_HC.iloc[motif]['Latent_Vector']
        latent_vector_BD = df_epoch_motif_BD.iloc[motif]['Latent_Vector']
        BD_HC_distance = dtw_ndim.distance(latent_vector_BD, latent_vector_HC)

    # Add the distance to the distances DataFrame
    distances_BD_HC_diagnal.append({'Epoch': epoch, 'Motif': motif, 'Distance': BD_HC_distance})

new_df = pd.DataFrame(distances_BD_HC_diagnal)
df = pd.concat([new_df, distances_HC_BD_df], ignore_index=True)
distances_HC_BD_df = df

distances_HC_BD_df.to_csv('distances_BD_HC_df.csv', index=False)
#%%
'''
Plot the pairwise DTW distances
'''
distances_HC_BD_df.replace([np.inf, -np.inf], 0, inplace=True)
distances_HC_HC_df.replace([np.inf, -np.inf], 0, inplace=True)
distances_BD_BD_df.replace([np.inf, -np.inf], 0, inplace=True)
# Calculate the mean distances for each epoch and motif
mean_distances_HC_BD = distances_HC_BD_df.groupby(['Epoch', 'Motif'])['Distance'].mean().reset_index()
mean_distances_HC_HC = distances_HC_HC_df.groupby(['Epoch', 'Motif'])['Distance'].mean().reset_index()
mean_distances_BD_BD = distances_BD_BD_df.groupby(['Epoch', 'Motif'])['Distance'].mean().reset_index()
sem_distances_HC_BD = distances_HC_BD_df.groupby(['Epoch', 'Motif'])['Distance'].sem().reset_index()
sem_distances_HC_HC = distances_HC_HC_df.groupby(['Epoch', 'Motif'])['Distance'].sem().reset_index()
sem_distances_BD_BD = distances_BD_BD_df.groupby(['Epoch', 'Motif'])['Distance'].sem().reset_index()
# Plotting
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

    ax.set_ylim([0, 2100])
    plt.show()

    pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_trajectory_distance'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "Motif{}-pairwise-dtw.png".format(motif)
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fname_pdf = "Motif{}-pairwise-dtw.pdf".format(motif)
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

#%% Plot the centroids in PC space in each epoch, in each motif.
labels_pop = list(label_all_)

state_epoch_centroids = []
state_epoch_volume = []
state_volume_ = []
len_latent_epoch = [0, 5 * 30 * 60, 10 * 30 * 60, 15 * 30 * 60]
len_latent = [0, len(Latent_vectors[0])]
count = 0
count_centroid = 0
for g in range(n_cluster):#10):
    epoch_centroids_per_subject = np.zeros((2, n_subject_in_population, 3)) # 2 pop x 25 subject each pop x 3 epoch
    epoch_centroids = [[], []]  # 2 x 10
    epoch_centroids_in_PC = [[], []]  # 2 x 3
    epoch_volume = [[], []]
    markers = ['o', '^', 'D']
    alphas = [0.4, 0.7, 1]

    idx_g = np.where(label_all_ == g)[0]
    latent_vec_all_epoch = latent_all_[idx_g, :]
    # old normalization method (not used)
    # nK_g = latent_vec_all_epoch.T @ latent_vec_all_epoch
    # volume_of_group = np.trace(nK_g)
    volume_of_group = np.trace(np.cov(latent_vec_all_epoch.T))
    state_volume_.append(volume_of_group)

    for epoch in range(1, 4):
        fig_pca_per_state = plt.figure(figsize=(6, 4))
        ax = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
        # for each population, get the centroid of each epoch
        data_to_plot = []
        label_to_plot = []
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

            # transformed_centroid = components_centroids[count, :]

            # # the centroids coordinate in PC space
            # epoch_centroids_in_PC[j].append(transformed_centroid)
            # # the centroid coordinate in latent space
            # epoch_centroids[j].append(centroids[count])
            count += 1
            data_to_plot.append(components[idx_g, :])
            label_to_plot.append(np.repeat(j, len(idx_g)))
        data_to_plot = np.vstack(data_to_plot)
        label_to_plot = np.hstack(label_to_plot)
        indices = list(range(len(label_to_plot)))
        # Shuffle the indices
        random.shuffle(indices)
        data_to_plot_shuffled = np.asarray([data_to_plot[i,:] for i in indices])
        label_to_plot_shuffled = [label_to_plot[i] for i in indices]
        colors = [cmap(g*2 + label) for label in label_to_plot_shuffled]
        ax.scatter3D(data_to_plot_shuffled[:,0], data_to_plot_shuffled[:,1], data_to_plot_shuffled[:,2],
                     # norm=plt.Normalize(vmin=0, vmax=9),
                     color=colors,
                     s=5,
                     alpha=0.3,
                     label='%d' % g, zorder=-1)

            # if len(latent_vec_g):
            #     #ng_K1 = latent_vec_g.T @ latent_vec_g volume_of_group_epoch = np.trace(ng_K1)
            #     volume_of_group_epoch = np.trace(np.cov(latent_vec_g.T))
            #     epoch_volume[j].append(volume_of_group_epoch)
            #     # Plot latent swarms
            #     ax.scatter3D(components[idx_g, 0], components[idx_g, 1], components[idx_g, 2],
            #               # norm=plt.Normalize(vmin=0, vmax=9),
            #               color=cmap(g * 2 + j),
            #               s=5,
            #               alpha=0.3,
            #               label='%d' % g, linewidth=2, zorder=-1)

        #     # plot centroid
        #     ax.scatter3D(transformed_centroid[0], transformed_centroid[1], transformed_centroid[2],
        #                  norm=plt.Normalize(vmin=0, vmax=9),
        #                  color=b_o_colors[j],
        #                  marker='.',
        #                  s=200,
        #                  label='{} epoch{} centroid'.format(titles[j], epoch), zorder=1)
        # # plot centroids of all subjects
        #     for sub in range(n_subject_in_population):
        #         sub_centroid = state_epoch_sub_centroids_PC[epoch - 1, j, sub, g, :]
        #         ax.scatter3D(sub_centroid[0], sub_centroid[1], sub_centroid[2],
        #                      norm=plt.Normalize(vmin=0, vmax=9),
        #                      color=b_o_colors[j],
        #                      marker='.',
        #                      s=100,
        #                      zorder=1)

        # plot distance between centroids between groups in same epoch
        # xs = [epoch_centroids_in_PC[0][epoch - 1][0], epoch_centroids_in_PC[1][epoch - 1][0]]
        # ys = [epoch_centroids_in_PC[0][epoch - 1][1], epoch_centroids_in_PC[1][epoch - 1][1]]
        # zs = [epoch_centroids_in_PC[0][epoch - 1][2], epoch_centroids_in_PC[1][epoch - 1][2]]
        # print(xs[0])
        # print(ys[0])
        # print(zs[0])
        # ax.plot3D(xs, ys, zs, linewidth=3, linestyle='-',
        #           color='k', label='epoch{}centroid between BD-HP'.format(epoch), zorder=1)

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




        # pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_centroid'.format(onedrive_path, project_name)
        # Path(pwd).mkdir(parents=True, exist_ok=True)
        # fname = "PCs of State {}-epoch{}-centroid.png".format(g, epoch)
        # fig_pca_per_state.savefig(os.path.join(pwd, fname), transparent=True)
        # fname_pdf = "PCs of State {}-epoch{}-centroid.pdf".format(g, epoch)
        # fig_pca_per_state.savefig(os.path.join(pwd, fname_pdf), transparent=True)
    # state_epoch_centroids.append(epoch_centroids)
    # state_epoch_volume.append(epoch_volume)
#%% Plot measures of centroids and volume

# inverse of the covariance matrix of the entire latent vector Z
iv = np.linalg.inv(np.cov(latent_all_.T))

for i in range(n_cluster):
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    #epoch_volume = state_epoch_volume[i]

    epoch_volume_list = [[],[]]
    epoch_volume_se_list = [[],[]]
    epoch_volume_normalize_list = [[], []]

    d_zit_t_minus_BD = [] # distance of motif i between time t (epoch 2), and time t-1 (epoch 1) in BD
    d_zit_t_minus_HP = []
    d_zit_t_minus_BD_new = []
    d_zit_t_minus_HP_new = []
    se_d_zit_t_minus_BD = []
    se_d_zit_t_minus_HP = []

    d_zit_BD_HP_ = np.empty((n_cluster, 3, n_subject_in_population * n_subject_in_population))
    d_zit_HP_HP_ = np.empty((n_cluster, 3, n_subject_in_population * n_subject_in_population))
    d_zit_BD_BD_ = np.empty((n_cluster, 3, n_subject_in_population * n_subject_in_population))
    # first, compute the distance between every BD subject to every HP subject in each epoch, in each motif
    for epoch in range(1,4):
        epoch_volume_BD = np.nanmean(person_volumes[epoch - 1, 1, :, i])  # state_epoch_volume[i] # 10 x 2 x 3
        epoch_volume_HP = np.nanmean(person_volumes[epoch - 1, 0, :, i])
        s_score = stats.ttest_ind(person_volumes[epoch - 1, 1, :, i], person_volumes[epoch - 1, 0, :, i], nan_policy='omit')

        # print("Motif{} Epoch{} latent volume, 2 sample t-stat: {:.2f}, p-val: {:.3f}".format(i, epoch, s_score.statistic, s_score.pvalue))

        epoch_volume_BD_se = np.nanstd(person_volumes[epoch - 1, 1, :, i]) / np.sqrt(
            n_subject_in_population)  # state_epoch_volume[i] # 10 x 2 x 3
        epoch_volume_HP_se = np.nanstd(person_volumes[epoch - 1, 0, :, i]) / np.sqrt(n_subject_in_population)

        d_zit_BD_HP_list = []
        d_zit_HP_HP_list = []
        d_zit_BD_BD_list = []
        for sub_i in range(n_subject_in_population):
            HP_person_centroid = person_centroids[epoch - 1, 0, sub_i, i, :]
            for sub_j in range(n_subject_in_population):
                BD_person_centroid = person_centroids[epoch - 1, 1, sub_j, i, :]
                if np.isnan(HP_person_centroid).all() or np.isnan(BD_person_centroid).all():
                    d_zit_BD_HP_between_two_person = nan
                else:
                    d_zit_BD_HP_between_two_person = np.linalg.norm(HP_person_centroid - BD_person_centroid)
                d_zit_BD_HP_list.append(d_zit_BD_HP_between_two_person)
            for sub_k in range(n_subject_in_population):
                HP_person_2_centroid = person_centroids[epoch - 1, 0, sub_k, i, :]
                if np.isnan(HP_person_centroid).all() or np.isnan(HP_person_2_centroid).all():
                    d_zit_HP_HP_between_two_person = nan
                else:
                    d_zit_HP_HP_between_two_person = np.linalg.norm(HP_person_centroid - HP_person_2_centroid)
                d_zit_HP_HP_list.append(d_zit_HP_HP_between_two_person)

        # compute the distance between every BD subject to every BD subject in each epoch, in each motif
        for sub_j in range(n_subject_in_population):
            BD_person_centroid = person_centroids[epoch - 1, 1, sub_j, i, :]
            for sub_l in range(n_subject_in_population):
                BD_person_2_centroid = person_centroids[epoch - 1, 1, sub_l, i, :]
                if np.isnan(BD_person_2_centroid).all() or np.isnan(BD_person_centroid).all():
                    d_zit_BD_BD_between_two_person = nan
                else:
                    d_zit_BD_BD_between_two_person = np.linalg.norm(BD_person_2_centroid - BD_person_centroid)
                d_zit_BD_BD_list.append(d_zit_BD_BD_between_two_person)

        d_zit_BD_HP_[i, epoch-1, : ] = d_zit_BD_HP_list
        d_zit_HP_HP_[i, epoch-1, : ] = d_zit_HP_HP_list
        d_zit_BD_BD_[i, epoch-1, : ] = d_zit_BD_BD_list
        print("Motif{} epoch {} distance between BD and HP centroids: {}".format(i, epoch, np.nanmean(d_zit_BD_HP_list)))
        # print("Motif{} epoch {} distance between HP and HP centroids: {}".format(i, epoch, np.nanmean(d_zit_HP_HP_list)))
        print("Motif{} epoch {} distance between BD and BD centroids: {}".format(i, epoch, np.nanmean(d_zit_BD_BD_list)))
        s_score = stats.ttest_ind(d_zit_BD_HP_list, d_zit_HP_HP_list, nan_policy='omit')
        # print("Motif{} Epoch{} d_zit_BD_HP -  HP-HP, 2 sample t-stat: {:.2f}, p-val: {}\n".format(i, epoch, s_score.statistic,
        #                                                                                    s_score.pvalue))
        s_score = stats.ttest_ind(d_zit_BD_HP_list, d_zit_BD_BD_list, nan_policy='omit')
        print("Motif{} Epoch{} d_zit_BD_BD - HP-BD, 2 sample t-stat: {:.2f}, p-val: {}\n".format(i, epoch, s_score.statistic,
                                                                                           s_score.pvalue))
        # then, compute the distance between every BD subject to same BD subject between t and t+ in each epoch, in each motif
        if epoch <= 2:
            d_BD = []
            d_HP = []
            for sub in range(25):
                if np.isnan(person_centroids[epoch - 1, 1, sub, i, :]).all() or np.isnan(person_centroids[epoch, 1, sub, i, :]).all():
                    d_BD.append(nan)
                else:
                    d_BD_zit_tplus = distance.mahalanobis(person_centroids[epoch - 1, 1, sub, i, :],
                                                          person_centroids[epoch, 1, sub, i, :],
                                                          iv)
                    d_BD.append(d_BD_zit_tplus)
            for sub in range(25):
                if np.isnan(person_centroids[epoch - 1, 0, sub, i, :]).all() or np.isnan(person_centroids[epoch, 0, sub, i, :]).all():
                    d_HP.append(nan)
                else:
                    dHP_zit_tplus = distance.mahalanobis(person_centroids[epoch - 1, 0, sub, i, :],
                                                         person_centroids[epoch, 0, sub, i, :],
                                                         iv)
                    d_HP.append(dHP_zit_tplus)

            d_zit_t_plus_BD_mean = np.nanmean(d_BD)# 3 epoch x 2 pop x 25 sub/pop x 10 stat x 3pc
            d_zit_t_plus_HP_mean = np.nanmean(d_HP)

            d_zit_t_plus_BD_se = np.nanstd(d_BD)/np.sqrt(25)
            d_zit_t_plus_HP_se = np.nanstd(d_HP)/np.sqrt(25)

            d_zit_t_minus_BD_new.append(d_zit_t_plus_BD_mean)
            d_zit_t_minus_HP_new.append(d_zit_t_plus_HP_mean)

            se_d_zit_t_minus_BD.append(d_zit_t_plus_BD_se)
            se_d_zit_t_minus_HP.append(d_zit_t_plus_HP_se)


        epoch_volume_list[0].append(epoch_volume_HP)
        epoch_volume_list[1].append(epoch_volume_BD)
        epoch_volume_se_list[0].append(epoch_volume_HP_se)
        epoch_volume_se_list[1].append(epoch_volume_BD_se)

        # epoch_volume_normalize_list[0].append(epoch_volume[0][epoch-1]/state_volume_[i])
        # epoch_volume_normalize_list[1].append(epoch_volume[1][epoch - 1]/state_volume_[i])



    d_this_epoch_zit_BD_HP = np.nanmean(d_zit_BD_HP_[i, :, : ], axis=1)
    d_this_epoch_zit_HP_HP = np.nanmean(d_zit_HP_HP_[i, :, :], axis=1)


    se_d_this_epoch_zit_BD_HP = np.nanstd(d_zit_BD_HP_[i, :, : ], axis=1)/np.sqrt(25*25)

    x = [0, 1, 2]
    x1 = [0.2, 1.2, 2.2]
    axes[0].errorbar(x, d_this_epoch_zit_BD_HP, yerr=se_d_this_epoch_zit_BD_HP, color='k', fmt='-o')
    axes[0].set_ylim(30, 60)
    axes[0].set_xticks(x)
    axes[0].set_title("State {} distance between BD and HP centroids".format(i))
    axes[0].grid(False)
    x2 = [0, 1]
    x3 = [0.2, 1.2]

    d_zit_t_minus_BD_new = np.asarray(d_zit_t_minus_BD_new)
    se_d_zit_t_minus_BD = np.asarray(se_d_zit_t_minus_BD)
    d_zit_t_minus_HP_new  = np.asarray(d_zit_t_minus_HP_new)
    se_d_zit_t_minus_HP = np.asarray(se_d_zit_t_minus_HP)
    axes[1].errorbar(x3, d_zit_t_minus_BD_new, yerr=se_d_zit_t_minus_BD, color=b_o_colors[1], fmt='--o')
    axes[1].errorbar(x2, d_zit_t_minus_HP_new, yerr=se_d_zit_t_minus_HP, color=b_o_colors[0], fmt='-o')
    axes[1].set_ylim(1, 4.5)
    axes[1].set_xticks(x2)
    axes[1].grid(False)
    axes[1].set_title("distance between centroids within population")

    axes[2].errorbar(x1, epoch_volume_list[0], yerr=epoch_volume_se_list[0], fmt='--o', label='HP', color=b_o_colors[0], markersize=10)
    axes[2].errorbar(x, epoch_volume_list[1], yerr=epoch_volume_se_list[1], fmt='-o', label='BD', color=b_o_colors[1], markersize=10)
    axes[2].set_ylim(0, 800)
    axes[2].set_xticks(x)
    axes[2].set_title("volume of BD and HP")
    axes[2].grid(False)

    # axes[3].plot(x, epoch_volume_normalize_list[0], '--o', label='BD', color=b_o_colors[1], markersize=10)
    # axes[3].plot(x, epoch_volume_normalize_list[1], '-o', label='HP', color=b_o_colors[0], markersize=10)
    # axes[3].set_xticks(x)
    # axes[3].set_title("relative volume portion/volume(state)")
    # axes[3].grid(False)

    fig.show()





    # pwd = r'{}\Behavior_VAE_data\{}\figure\PCA_visual\epoch_centroid'.format(onedrive_path, project_name)
    # Path(pwd).mkdir(parents=True, exist_ok=True)
    # fname = "State {}-centroid-distance-v2.png".format(i)
    # fig.savefig(os.path.join(pwd, fname), transparent=True)
    # fname_pdf = "State {}-centroid-distance-v2.pdf".format(i)
    # fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

#%% Statistical tests

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
