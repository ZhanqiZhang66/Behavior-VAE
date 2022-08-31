# Created by Victoria Zhang at 4/26/2022
# File: latent_visualization.py
# Description: visualize latent variable from latent space VAME
# Scenario:
# Usage:
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
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
Epoch1_motif_usage = [[], []]
Epoch1_latent_vector = [[], []]
Epoch1_transition_matrix = [[], []]
Epoch1_motif_usage_cat = [[],[]]
Epoch1_Entropies = [[],[]]
Epoch1_num_zero_rows = [[],[]]
Epoch1_num_zeros = [[],[]]
Epoch1_num_ones = [[],[]]
Epoch1_volume_per_person = [[],[]] # trace of cov(latent vector of this person)


Epoch2_labels = [[], []]
Epoch2_motif_usage = [[], []]
Epoch2_latent_vector = [[], []]
Epoch2_transition_matrix = [[], []]
Epoch2_motif_usage_cat = [[],[]]
Epoch2_Entropies = [[],[]]
Epoch2_num_zero_rows = [[],[]]
Epoch2_num_zeros = [[],[]]
Epoch2_num_ones = [[],[]]
Epoch2_volume_per_person = [[],[]] # trace of cov(latent vector of this person)

Epoch3_labels = [[], []]
Epoch3_motif_usage = [[], []]
Epoch3_latent_vector = [[], []]
Epoch3_transition_matrix = [[], []]
Epoch3_motif_usage_cat = [[],[]]
Epoch3_Entropies = [[],[]]
Epoch3_num_zero_rows = [[],[]]
Epoch3_num_zeros = [[],[]]
Epoch3_num_ones = [[],[]]
Epoch3_volume_per_person = [[],[]] # trace of cov(latent vector of this person)

Entropies = [[],[]]
num_zero_rows = [[],[]]
num_zeros = [[],[]]
num_ones = [[],[]]

#%%
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
        Latent_len[j].append(len(latent_vector))

        # variance of each person
        K = np.cov(latent_vector.T)  # covariance matrix of the 10-D latent vector
        volume = np.trace(K)
        volume_per_person[j].append(volume)
        transition_matrices.append(transition_m)

        v_index = start_frame.loc[start_frame['video_name'] == v].index.values[0]
        door_close_time = start_frame.loc[v_index, 'door_close']
        start_time = start_frame.loc[v_index, 'n']
        five_min_frame_no = int(5 * 60 * 30)
        offset = int(start_time - door_close_time)

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
#%% Population-wise plot
#%% plot PCA for each video, all states
titles = ["CP", "BD"]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Computing {} ...".format(v))
        label = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(project_name, v,n_cluster,n_cluster,v))
        cluster_center = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 10

        pca = PCA(n_components=3)
        components = pca.fit_transform(latent_vector)
        components_cluster_center = pca.fit_transform(cluster_center)
        total_var = pca.explained_variance_ratio_.sum() * 100
        principalDf = pd.DataFrame(data=components)
        finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
        finalDf.columns = ['pc 1', 'pc 2', 'pc 3','target']

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # 3D PCA
        ax = fig.add_subplot(1,1,1,projection='3d')
        t = np.arange(10)
        for g in np.unique(label):
            i = np.where(label == g)
            cmap = plt.cm.Spectral
            ax.scatter(components[i, 0], components[i, 1], components[i, 2], norm=plt.Normalize(vmin=0, vmax=9),c=label[i],cmap='Spectral', s=2, alpha=0.05, label='%d' % g)

        ax.set_title("PCs of {}-{}\n Exp_Var:{:.2f}".format(titles[j], v, total_var))
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_xlim(-55, 55)
        ax.set_ylim(-55, 55)
        ax.set_zlim(-55, 55)
        fig.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
        fname = "PCs of {}-{}-3d.png".format(titles[j], v)
        fig.savefig(os.path.join(pwd, fname))

        # 2D PCA
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for g in np.unique(label):
            i = np.where(label == g)
            cmap = plt.cm.Spectral
            ax.scatter(components[i, 0], components[i, 1], norm=plt.Normalize(vmin=0, vmax=9),
                          c=label[i], cmap='Spectral', s=2, alpha=0.05, label='%d' % g)

        ax.set_title("PCs of {}-{}\n Exp_Var:{:.2f}".format(titles[j], v, pca.explained_variance_ratio_[:-2].sum() * 100))
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_xlim(-55, 55)
        ax.set_ylim(-55, 55)
        ax.set_aspect('equal', 'box')

        fig.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
        fname = "PCs of {}-{}-2d.png".format(titles[j], v)
        fig.savefig(os.path.join(pwd, fname))
#%% Plot PCA of BD and CP population, for each state
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
fig_pca = plt.figure(figsize=(10,10))
fig_latent = plt.figure(figsize=(10,10))

pca = PCA(n_components=3)
K_var = np.zeros((10, 2))
K_var_all_subjects = np.zeros((12, 10, 2))
for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    latent_vec = Latent_vectors[j]
    latent_vec_trim = latent_vec
    label = Labels[j]
    label_trim = Labels[j]
    ax = fig_pca.add_subplot(2, 1, j+1,  projection='3d')
    for sub in range(12):
        latent_vec_sub = latent_vec_trim[0: Latent_len[j][sub]]
        latent_vec_trim = latent_vec_trim[Latent_len[j][sub]:]

        label_sub = label_trim[0: Latent_len[j][sub]]
        label_trim = label_trim[Latent_len[j][sub]:]
        sub_name = videos[sub]
        for g in np.unique(label):
            #  state-population-subject
            i_sub = np.where(label_sub == g)
            if len(i_sub[0]):
                latent_sub_g = latent_vec_sub[i_sub]
                components = pca.fit_transform(latent_sub_g)
                total_var = pca.explained_variance_ratio_.sum() * 100
                principalDf = pd.DataFrame(data=components)
                finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
                finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
                K_sub = np.cov(latent_sub_g.T)
                volume_of_group_sub = np.trace(K_sub)
                K_var_all_subjects[sub][g][j] = volume_of_group_sub
            else:
                volume_of_group_sub = 0
                K_var_all_subjects[sub][g][j] = volume_of_group_sub
            fig_pca_per_state = plt.figure(figsize=(10, 10))
            ax2 = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')

            ax.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
                       color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
            ax2.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
                        color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)

            ax2.set_title("PCs of {}-{}-State- {}\n Exp_Var:{:.2f}".format(titles[j], sub_name, g, total_var))
            ax2.set_xlabel('PC 1')
            ax2.set_ylabel('PC 2')
            ax2.set_zlabel('PC 3')
            ax2.set_xlim(-50, 50)
            ax2.set_ylim(-50, 50)
            ax2.set_zlim(-50, 50)
            fig_pca_per_state.show()
            pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
            fname = "PCs of {}-{} State {}.png".format(titles[j], sub_name, g)
            fig_pca_per_state.savefig(os.path.join(pwd, fname))


            #  state-population wise
        #     i = np.where(label == g)
        #     latent_g = latent_vec[i]
        #     components = pca.fit_transform(latent_g)
        #     total_var = pca.explained_variance_ratio_.sum() * 100
        #     principalDf = pd.DataFrame(data=components)
        #     finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
        #     finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
        #     K = np.cov(latent_g.T)
        #     volume_of_group = np.trace(K)
        #     K_var[g][j] = volume_of_group
        #     fig_pca_per_state = plt.figure(figsize=(10, 10))
        #     ax2 = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
        #
        #     ax.scatter(components[:, 0], components[:, 1], components[:, 2],norm=plt.Normalize(vmin=0, vmax=9),
        #                       color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
        #     ax2.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
        #                color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
        #     ax2.set_title("PCs of {}-State {}\n Exp_Var:{:.2f}".format(titles[j], g, total_var))
        #     ax2.set_xlabel('PC 1')
        #     ax2.set_ylabel('PC 2')
        #     ax2.set_zlabel('PC 3')
        #     ax2.set_xlim(-50, 50)
        #     ax2.set_ylim(-50, 50)
        #     ax2.set_zlim(-50, 50)
        #     fig_pca_per_state.show()
        #     pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
        #     fname = "PCs of {} State {}.png".format(titles[j], g)
        #     fig_pca_per_state.savefig(os.path.join(pwd, fname))
        #
        # ax.set_title("PCs of {}-\n Exp_Var:{:.2f}".format(titles[j], total_var))
        # ax.set_xlabel('PC 1')
        # ax.set_ylabel('PC 2')
        # ax.set_zlabel('PC 3')
        # ax.set_xlim(-50, 50)
        # ax.set_ylim(-50, 50)
        # ax.set_zlim(-50, 50)
        # fig_pca.show()
        # pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
        # fname = "PCs of BD-CP.png"
        # fig_pca.savefig(os.path.join(pwd, fname))
        # plt.close('all')
#
#%%

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
x = np.arange(10)
ax.plot(x, K_var[:,0], '-o')
ax.plot(x, K_var[:, 1],'-o')
ax.set_title('Volume of state')
ax.set_xticks(x)
ax.set_xlabel('State')
ax.set_ylabel('volume of state')
ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
fig.show()
# pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual\epoch'
# fname = "Volume of state {}.png".format(g)
# fig.savefig(os.path.join(pwd, fname))
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


ani = FuncAnimation(fig, animation, frames = 240, interval = 500, blit = True)

plt.show()
#%%
for j, videos in enumerate([control_videos, BD_videos]):
    color = 'C{}'.format(j)

    for sub in range(12):
        sub_name = videos[sub]
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = np.arange(10)
        ax.scatter(x, K_var_all_subjects[sub,:,j], c=color)
        ax.set_title('{} Volume of state'.format(sub_name))
        ax.set_xticks(x)
        ax.set_xlabel('State')
        ax.set_ylabel('volume of state')
        ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
        fig.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
        fname = "{}_{}_volumn.png".format(sub_name, n_cluster)
        fig.savefig(os.path.join(pwd, fname))

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
x = np.arange(10)
for sub in range(12):
    ax.scatter(x, K_var_all_subjects[sub, :, 0], c='C0', alpha=0.5)
    ax.scatter(x+0.2, K_var_all_subjects[sub, :, 1], c='C1', alpha=0.5)
    x = np.arange(10)

ax.plot(x, np.median(K_var_all_subjects[:, :, 0],axis=0), '-^', color='C0')
ax.plot(x, np.median(K_var_all_subjects[:, :, 1],axis=0),'-^', color='C1')
ax.set_title('Volume of state')
ax.set_xticks(x)
ax.set_xlabel('State')
ax.set_ylabel('volume of state')
ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
fig.show()
pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
fname = "{}_volume.png".format(n_cluster)
fig.savefig(os.path.join(pwd, fname))
#%% Epoch-wise plot




#%%
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
pca = PCA(n_components=3)
K_var = [np.zeros((10, 3)), np.zeros((10, 3))]
K_var_all_subjects = [np.zeros((12,10, 3)), np.zeros((12, 10, 3))]
for j, videos in enumerate([control_videos, BD_videos]):
    for epoch in range(1, 4):
        latent_vec = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        label = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        components = pca.fit_transform(latent_vec)
        total_var = pca.explained_variance_ratio_.sum() * 100
        principalDf = pd.DataFrame(data=components)
        finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
        finalDf.columns = ['pc 1', 'pc 2', 'pc 3', 'target']
        for sub in range(12):
            latent_vec_sub = latent_vec_trim[0: Latent_len_epoch[epoch-1][j][sub]]
            latent_vec_trim = latent_vec_trim[Latent_len_epoch[epoch-1][j][sub]:]

            label_sub = label_trim[0: Latent_len[j][sub]]
            label_trim = label_trim[Latent_len[j][sub]:]
            sub_name = videos[sub]
            for g in np.unique(label):
                #state-subject-plot
                #  state-population-subject
                i_sub = np.where(label_sub == g)
                if len(i_sub[0]):
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
                    #TODO K_var_all_subjects is all zeros
                K_temp = K_var_all_subjects[j]
                K_temp[sub][g][epoch - 1] = volume_of_group_sub
                K_var_all_subjects[j] = K_temp
                # fig_pca_per_state = plt.figure(figsize=(10, 10))
                # ax2 = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
                #
                # ax.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
                #            color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
                # ax2.scatter(components[:, 0], components[:, 1], components[:, 2], norm=plt.Normalize(vmin=0, vmax=9),
                #             color=cmap(g * 2 + j), s=2, alpha=0.05, label='%d' % g)
                #
                # ax2.set_title("PCs of {}-{}-State- {}\n Exp_Var:{:.2f}".format(titles[j], sub_name, g, total_var))
                # ax2.set_xlabel('PC 1')
                # ax2.set_ylabel('PC 2')
                # ax2.set_zlabel('PC 3')
                # ax2.set_xlim(-50, 50)
                # ax2.set_ylim(-50, 50)
                # ax2.set_zlim(-50, 50)
                # fig_pca_per_state.show()
                # pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
                # fname = "PCs of {}-{} State {}.png".format(titles[j], sub_name, g)
                # fig_pca_per_state.savefig(os.path.join(pwd, fname))

                # state-population plot
                i = np.where(label == g)
                latent_vec_g = latent_vec[i]
                if len(latent_vec_g):
                    K = np.cov(latent_vec_g.T)
                    volume_of_group = np.trace(K)
                else:
                    volume_of_group = 0
                K_var[j][g][epoch-1] = volume_of_group
                # fig_pca_per_state = plt.figure(figsize=(10, 10))
                # ax2 = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
                #
                # ax.scatter(components[i, 0], components[i, 1], components[i, 2],norm=plt.Normalize(vmin=0, vmax=9),
                #                   color=cmap(g * 2 + j), s=3, alpha=0.1, label='%d' % g)
                # ax2.scatter(components[i, 0], components[i, 1], components[i, 2], norm=plt.Normalize(vmin=0, vmax=9),
                #            color=cmap(g * 2 + j), s=3, alpha=0.1, label='%d' % g)
                #
                # ax2.set_title("PCs of {}-State-{}-Epoch-{} \n volume:{:.2f}".format(titles[j], g,epoch, volume_of_group))
                # ax2.set_xlabel('PC 1')
                # ax2.set_ylabel('PC 2')
                # ax2.set_zlabel('PC 3')
                # ax2.set_xlim(-50, 50)
                # ax2.set_ylim(-50, 50)
                # ax2.set_zlim(-50, 50)
                # fig_pca_per_state.show()
                # pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual\epoch'
                # fname = "PCs of {} State {} Epoch {}.png".format(titles[j], g, epoch)
                # fig_pca_per_state.savefig(os.path.join(pwd, fname))

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
    pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual\epoch'
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
            pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
            fname = "{}_{}_epoch{}_volumn.png".format(sub_name, n_cluster, epoch+1)
            fig.savefig(os.path.join(pwd, fname))
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
    pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\PCA_visual'
    fname = "{}_volume_epoch_{}.png".format(n_cluster, epoch)
    fig.savefig(os.path.join(pwd, fname))
