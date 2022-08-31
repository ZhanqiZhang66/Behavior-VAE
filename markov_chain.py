# Created by Victoria Zhang at 7/24/2022
# File: markov_chain.py
# Description: Plot transition matrix, latent visualization of the markov chain
# Scenario:
# Usage:

import umap
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.decomposition import PCA

import os
import scipy
from vame.analysis.community_analysis import  read_config#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
#%%
project_name = 'BD20-Jun5-2022'
config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
n_cluster = 10
model_name = 'VAME'
control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
n_video_per_group = len(control_videos)
N = [0,0]
Motif_usages = [[],[]]
Entropies = [[],[]]
num_zero_rows = [[],[]]
num_zeros = [[],[]]
num_ones = [[],[]]
motif_usage_cat = [[],[]]
Latent_vectors = [[],[]]
Latent_vectors_per_state = [[],[]] # mean and variance of state 0,1,...9 of every person's latent vector
State_vectors_per_latent_d = [[],[]] # mean and variance of latent d = 0,1,...9 of every person's state
Labels = [[], []]
TM = [[], []]
volume_per_person = [[],[]] # trace of cov(latent vector of this person)
Stats_per_latent_per_state = [[],[]]
Cluster_center = []
transition_matrices = []

def add_self_transition(transition_m):
    transition = transition_m.copy()
    zero_rows = np.all(transition == 0, axis=1)
    zero_rows_i =  np.where(zero_rows == True)
    zero_cols = np.all(transition == 0, axis=0)
    zero_cols_i = np.where(zero_cols == True)

    # add self transition
    if np.sum(zero_rows) != np.sum(zero_cols):
        self_transition_i = list(set(zero_rows_i[0]) ^ set(zero_cols_i[0]))
        for i in self_transition_i:
            transition[i,i] = 1
    if zero_rows_i[0].size != 0 or zero_cols_i[0].size != 0: # when there are rows or zeros, or colums of zeros
        zeros_rows_colums_i = list(set(zero_rows_i[0]) & set(zero_cols_i[0])) # remove them
        idx_to_keep = np.ones(len(transition_m), dtype=bool)
        for i in range(len(transition_m)):
            if i in zeros_rows_colums_i:
                idx_to_keep[i] = False
        transition = transition[idx_to_keep]
        transition = transition[:, idx_to_keep]
    return transition, len(zero_cols_i[0]),  np.count_nonzero(transition == 1), np.count_nonzero(transition==0)
titles = ["CP", "BD"]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Computing {} ...".format(v))
        label = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(project_name, v,n_cluster,n_cluster,v))
        transition_m = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(project_name, v,n_cluster, v))
        cluster_center = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(project_name, v,n_cluster, v))
        motif_usage = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 10

        # variance of each person
        K = np.cov(latent_vector.T) # covariance matrix of the 10-D latent vector
        volume = np.trace(K)
        volume_per_person[j].append(volume)
        transition_matrices.append(transition_m)

        # Entropy, and classic metrics
        # https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
        invertible_T, num_zero_row, num_one_item, num_zero_item = add_self_transition(transition_m)
        S, U = scipy.linalg.eig(invertible_T.T)
        stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        stationary = stationary / np.sum(stationary)
        stationary = stationary.real
        H = scipy.stats.entropy(invertible_T, base=2)
        entropy = stationary.dot(H)
        if np.isnan(entropy):
            break
        Entropies[j].append(entropy)
        num_state = len(invertible_T) + 1
        num_zero_rows[j].append(num_zero_row/num_state)
        num_ones[j].append(num_one_item/(num_state*num_state))
        num_zeros[j].append(num_zero_item/(num_state*num_state))

        # reformat motif usage & transition matrix
        if i == 0:
            l = label
            latent = latent_vector
            m = motif_usage
            tm = transition_m
            Cluster_center.append(cluster_center)
        else:
            latent = np.concatenate([latent, latent_vector])
            l = np.concatenate([l,label])
            m += motif_usage
            tm += transition_m
            Cluster_center.append(cluster_center)
        motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
        num_points = latent_vector.shape[0]
        n += num_points


        latent_variance_per_state = []
        stats_per_latent_per_state = np.empty((n_cluster, np.shape(latent_vector)[1], 2)) # 10 x 10
        for state_i in range(n_cluster):
            state_i_idx = label == state_i
            latent_state_i = latent_vector[state_i_idx, :] #time of (state i) x 10
            K_per_state = np.cov(latent_state_i.T)
            latent_variance_per_state.append([np.mean(latent_state_i), np.trace(K_per_state)])
            for latent_d in range(np.shape(latent_vector)[1]):
                latent_d_state_i = latent_state_i[:, latent_d]
                stats_per_latent_per_state[state_i, latent_d, 0] = np.nanmean(latent_d_state_i)
                stats_per_latent_per_state[state_i, latent_d, 1] = np.nanvar(latent_d_state_i)

        Stats_per_latent_per_state[j].append(stats_per_latent_per_state)
        Latent_vectors_per_state[j].append(latent_variance_per_state)

        # state_variance_per_latent = []
        # for latent_d in range(np.shape(latent_vector)[1]):
        #     latent_d_vector = latent_vector[:, latent_d]
        #     K_per_latent = np.cov(latent_d_vector)
        #     state_variance_per_latent.append([np.mean(latent_d_vector), np.trace(K_per_latent)])
        # State_vectors_per_latent_d[j].append(state_variance_per_latent)

    N[j] = n
    Motif_usages[j] = m
    Latent_vectors[j] = latent
    Labels[j] = l
    TM[j] = tm/12

#%%                     PLOT







#%% Plot spasity, entropy, number of 1s

fig, axes = plt.subplots(4, figsize=(5, 10))
sns.set_style('darkgrid')
s1 = sns.distplot(Entropies[0], ax=axes[0])
s2 = sns.distplot(Entropies[1], ax=axes[0])
axes[0].set_title('entropy')
axes[0].legend( ['CP','BD'])

sns.distplot(num_zero_rows[0], ax=axes[1])
sns.distplot(num_zero_rows[1], ax=axes[1])
axes[1].set_title('#zero state')

sns.distplot(num_ones[0], ax=axes[2])
sns.distplot(num_ones[1], ax=axes[2])
axes[2].set_title('%p=1')

sns.distplot(num_zeros[0], ax=axes[3])
sns.distplot(num_zeros[1], ax=axes[3])
axes[3].set_title('% p=0')

plt.show()
#%% latent vector population violin
CPstate_IDs = np.meshgrid(np.arange(Latent_vectors[0].shape[1]),np.arange(Latent_vectors[0].shape[0]))[0].flatten()
BDstate_IDs = np.meshgrid(np.arange(Latent_vectors[1].shape[1]),np.arange(Latent_vectors[1].shape[0]))[0].flatten()
CP_idx = np.zeros(len(CPstate_IDs))
BD_idx = np.ones(len(BDstate_IDs))
latent_CP = Latent_vectors[0].flatten()
latent_BD = Latent_vectors[1].flatten()

latent_ds = pd.DataFrame(np.concatenate((
    np.concatenate((CPstate_IDs, BDstate_IDs),0).reshape(-1, 1),
     np.concatenate((latent_CP, latent_BD),0).reshape(-1, 1),
    np.concatenate((CP_idx, BD_idx),0).reshape(-1, 1)), 1),
    columns=['states', 'latent', 'is_BD'])

fig, ax = plt.subplots(1, figsize=(10, 5))
# latent_ds_CP = pd.DataFrame(Latent_vectors[0])
sns.violinplot(x="states", y="latent", hue="is_BD",
            data=latent_ds, split=True, scale="count", palette="muted")
ax.set_xlabel('States (Motifs)')
plt.show()
#%% plot PCA and latent vector on latent space (1,2,3)
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


        fig = plt.figure(figsize=(12,10))
        # 3D PCA
        ax = fig.add_subplot(2,2,1,projection='3d')
        t = np.arange(10)
        for g in np.unique(label):
            i = np.where(label == g)
            cmap = plt.cm.Spectral
            ax.scatter(components[i, 0], components[i, 1], components[i, 2], norm=plt.Normalize(vmin=0, vmax=9),c=label[i],cmap='Spectral', s=2, alpha=0.05, label='%d' % g)

        ax.set_title("PCs of {}-{}\n Exp_Var:{:.2f}".format(titles[j], v, total_var))
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        # legend1 = ax.legend(*scatter.legend_elements(),
        #                     loc="lower left", title="States/Motifs", bbox_to_anchor=(1, 0.5))
        # ax.add_artist(legend1)

        # 2D PCA
        ax = fig.add_subplot(2,2,2)
        for g in np.unique(label):
            i = np.where(label == g)
            cmap = plt.cm.Spectral
            ax.scatter(components[i, 0], components[i, 1], norm=plt.Normalize(vmin=0, vmax=9),
                          c=label[i], cmap='Spectral', s=2, alpha=0.05, label='%d' % g)

        ax.set_title("PCs of {}-{}\n Exp_Var:{:.2f}".format(titles[j], v, pca.explained_variance_ratio_[:-2].sum() * 100))
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal', 'box')

        # 3D latent vector
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        for g in np.unique(label):
            i = np.where(label == g)
            ax.scatter(latent_vector[i, 0], latent_vector[i, 1], latent_vector[i, 2], norm=plt.Normalize(vmin=0, vmax=9),
                       c=label[i], cmap='Spectral', s=2, alpha=0.05, label='%d' % g)

        scatter = ax.scatter(cluster_center[:, 0], cluster_center[:, 1],
                             norm=plt.Normalize(vmin=0, vmax=9), c=t, cmap='Spectral', s=80, alpha=1.,
                             edgecolors="black", label=t)
        ax.set_title(
            "latent vector of {}-{}".format(titles[j], v))
        ax.set_xlabel('L 1')
        ax.set_ylabel('L 2')
        ax.set_zlabel('L 3')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)

        # 2D latent vector
        ax = fig.add_subplot(2, 2, 4)
        for g in np.unique(label):
            i = np.where(label == g)
            cmap = plt.cm.Spectral
            ax.scatter(latent_vector[i, 0], latent_vector[i, 1], norm=plt.Normalize(vmin=0, vmax=9),
                       c=label[i], cmap='Spectral', s=2, alpha=0.05, label='%d' % g)

        scatter = ax.scatter(cluster_center[:, 0], cluster_center[:, 1],
                             norm=plt.Normalize(vmin=0, vmax=9), c=t, cmap='Spectral', s=80, alpha=1.,
                             edgecolors="black", label=t)
        ax.set_title(
            "latent vector of {}-{}".format(titles[j], v))
        ax.set_xlabel('L 1')
        ax.set_ylabel('L 2')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect('equal', 'box')
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="States/Motifs", bbox_to_anchor=(1, 0.5))
        ax.add_artist(legend1)

        plt.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure'
        fname = "PCs of {}-{}.png".format(titles[j], v)
        plt.savefig(os.path.join(pwd, fname))


#%% Plot how much volume each do person occupy in latent space?
colors = np.array(["C0", "C1"])
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Computing {} ...".format(v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 10

        fig_person = plt.figure(figsize=(10,10))
        ax = fig_person.add_subplot(1, 1, 1, projection='3d')
        latent_len = np.shape(latent_vector)[0]
        z = np.full((1, latent_len), colors[j])
        ax.scatter(latent_vector[:, 0], latent_vector[:, 1], latent_vector[:, 2], s=2,c=colors[j], alpha=0.05)
        ax.set_title("latent vectors of {}-{}, Volume %{:.2f}".format(titles[j], v,
                                                                              volume_per_person[j][n] * 100 / np.sum(
                                                                                  volume_per_person[j])))
        ax.set_xlabel('L 1')
        ax.set_ylabel('L 2')
        ax.set_zlabel('L 3')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        fig_person.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'.format()
        os.makedirs(pwd, exist_ok=True)
        fname = "latent vectors of {}-{}.png".format(titles[j], v,)
        fig_person.savefig(os.path.join(pwd, fname))
        n += 1
plt.close('all')
#%% population volume plot
fig_person = plt.figure(figsize=(10,10))
ax = fig_person.add_subplot(1, 1, 1)
sns.distplot(volume_per_person[0], ax=ax)
sns.distplot(volume_per_person[1], ax=ax)
ax.legend( ['CP','BD'])
ax.set_title('Trace of Latent Vector')
pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure'
fname = "Volume per group.png"
fig_person.savefig(os.path.join(pwd, fname))
#%% Plot latent vector, and PCA of BD and CP population

titles = ["CP", "BD"]
matplotlib.use('Qt5Agg')

fig_pca = plt.figure(figsize=(10,10))
fig_latent = plt.figure(figsize=(10,10))

pca = PCA(n_components=3)
for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    motif = Motif_usages[j]
    latent_vec = Latent_vectors[j]
    label = Labels[j]
    K = np.cov(latent_vec.T)
    volume_of_group = np.trace(K)


    # tsne_embed = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(latent[:n, :])
    fig_latent_per_state = plt.figure(figsize=(10, 10))
    ax = fig_latent.add_subplot(2, 1, j + 1, projection='3d')

    t = np.arange(10)
    volume_of_state = 0

    for g in np.unique(label):
        state_i_idx = label == g
        latent_state_i = latent_vec[state_i_idx, :]  # time of (state i) x 10
        K_per_state = np.cov(latent_state_i.T)
        volume_of_group_state = np.trace(K_per_state)


        i = np.where(label == g)
        fig_latent_per_state = plt.figure(figsize=(10, 10))

        ax.scatter(latent_vec[i, 0], latent_vec[i, 1], latent_vec[i, 2],norm=plt.Normalize(vmin=0, vmax=9),
                   c=label[i], cmap='Spectral', s=2, alpha=0.05)

        # latent vector at each state
        cmap = plt.get_cmap('tab20b')
        ax1 = fig_latent_per_state.add_subplot(1, 1, 1, projection='3d')
        ax1.scatter(latent_vec[i, 0], latent_vec[i, 1], latent_vec[i, 2],
                   c=cmap(g*2+j), s=2, alpha=0.1, label=titles[j])
        ax1.legend()

        scatter1 = ax1.scatter(cluster_center[:, 0], cluster_center[:, 1], cluster_center[:, 2],
                             norm=plt.Normalize(vmin=0, vmax=9),
                             c=t, cmap='Spectral', s=80, alpha=1., edgecolors="black", label=t)  # cluster center
        legend2 = ax1.legend(*scatter1.legend_elements(),
                            loc="lower left", title="States/Motifs", bbox_to_anchor=(1, 0.5))
        ax1.add_artist(legend2)
        ax1.set_title("latent vectors of {}, State {}, Volume %{:.2f}".format(titles[j], g, volume_of_group_state*100/volume_of_group))
        ax1.set_xlabel('L 1')
        ax1.set_ylabel('L 2')
        ax1.set_zlabel('L 3')
        ax1.set_xlim(-50, 50)
        ax1.set_ylim(-50, 50)
        ax1.set_zlim(-50, 50)
        fig_latent_per_state.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'.format()
        os.makedirs(pwd, exist_ok=True)
        fname = "Latents of {} State {}.png".format(titles[j], g)
        fig_latent_per_state.savefig(os.path.join(pwd, fname))
        plt.close('all')


    scatter = ax.scatter(cluster_center[:, 0], cluster_center[:, 1], cluster_center[:, 2],norm=plt.Normalize(vmin=0, vmax=9),
                          c=t, cmap='Spectral',s=80, alpha=1.,edgecolors= "black", label=t)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="States/Motifs", bbox_to_anchor=(1, 0.5))
    ax.add_artist(legend1)
    # ax.legend(labels=t, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("latent vectors  of {}".format(titles[j]))
    ax.set_xlabel('L 1')
    ax.set_ylabel('L 2')
    ax.set_zlabel('L 3')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)


    ax = fig_pca.add_subplot(2, 1, j+1,  projection='3d')
    # num = np.unique(l)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_title("UMAP of {}".format(titles[j]))

    # PCA
    components = pca.fit_transform(latent_vec[:n, :])
    total_var = pca.explained_variance_ratio_.sum() * 100
    principalDf = pd.DataFrame(data=components)
    finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
    finalDf.columns = ['pc 1', 'pc 2','pc 3','target']
    # ax = fig.add_subplot(3, 2, j+5)
    for g in np.unique(label):
        fig_pca_per_state = plt.figure(figsize=(10, 10))
        ax2 = fig_pca_per_state.add_subplot(1, 1, 1, projection='3d')
        i = np.where(label == g)
        ax.scatter(components[i, 0], components[i, 1], components[i, 2],norm=plt.Normalize(vmin=0, vmax=9),
                          c=label[i], cmap='Spectral', s=2, alpha=0.05, label='%d' % g)
        ax2.scatter(components[i, 0], components[i, 1], components[i, 2], norm=plt.Normalize(vmin=0, vmax=9),
                   c=label[i], cmap='Spectral', s=2, alpha=0.05, label='%d' % g)

        ax2.set_title("PCs of {}-State {}\n Exp_Var:{:.2f}".format(titles[j], g, total_var))
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        ax2.set_zlabel('PC 3')
        ax2.set_xlim(-50, 50)
        ax2.set_ylim(-50, 50)
        ax2.set_zlim(-50, 50)
        fig_pca_per_state.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'
        fname = "PCs of {} State {}.png".format(titles[j], g)
        fig_pca_per_state.savefig(os.path.join(pwd, fname))
    ax.set_title("PCs of {}-\n Exp_Var:{:.2f}".format(titles[j], total_var))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 50)
    fig_pca.show()
    pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'
    fname = "PCs of BD-CP.png"
    fig_pca.savefig(os.path.join(pwd, fname))
    plt.close('all')



fig_pca.show()
pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'
fname = "PCs of BD-CP.png"
fig_pca.savefig(os.path.join(pwd, fname))

fig_latent.show()
pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'
fname = "Latents of BD-CP.png"
fig_latent.savefig(os.path.join(pwd, fname))
plt.close('all')
#%% volume of every state, separated by population
titles = ["CP", "BD"]
matplotlib.use('Qt5Agg')
cmap = plt.get_cmap('tab20')
t = np.arange(10)

for g in range(n_cluster):
    fig_latent_per_state = plt.figure(figsize=(10, 10))
    ax1 = fig_latent_per_state.add_subplot(1, 1, 1, projection='3d')
    volume_each_state = []
    for j, videos in enumerate([control_videos, BD_videos]):
        n = N[j]
        motif = Motif_usages[j]
        latent_vec = Latent_vectors[j]
        label = Labels[j]
        K = np.cov(latent_vec.T)
        volume_of_group = np.trace(K)

        state_i_idx = label == g
        latent_state_i = latent_vec[state_i_idx, :]  # time of (state i) x 10
        K_per_state = np.cov(latent_state_i.T)
        volume_of_group_at_state = np.trace(K_per_state)
        volume_each_state.append(volume_of_group_at_state)
        i = np.where(label == g)

        ax1.scatter(latent_vec[i, 0], latent_vec[i, 1], latent_vec[i, 2],
                    color=cmap(g * 2 + j), s=4, alpha=0.02, label=titles[j])
    lgnd = ax1.legend()
    for lh in lgnd.legendHandles:
        lh.set_alpha(1)
        lh._sizes = [10]


    # cluster center
    # scatter1 = ax1.scatter(cluster_center[:, 0], cluster_center[:, 1], cluster_center[:, 2],
    #                        norm=plt.Normalize(vmin=0, vmax=9),
    #                        c=t, cmap='Spectral', s=80, alpha=1., edgecolors="black", label=t)  # cluster center
    # legend2 = ax1.legend(*scatter1.legend_elements(),
    #                      loc="lower left", title="States/Motifs", bbox_to_anchor=(1, 0.5))
    # ax1.add_artist(legend2)
    ax1.set_title("latent vectors of BD, CP populations, State {}, Volume {:.2f}-vs-{:.2f}".format(g,
                                                                          volume_each_state[0], volume_each_state[1]))
    ax1.set_xlabel('L 1')
    ax1.set_ylabel('L 2')
    ax1.set_zlabel('L 3')
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.set_zlim(-50, 50)
    fig_latent_per_state.show()
    pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'.format()
    os.makedirs(pwd, exist_ok=True)
    fname = "latent vectors of BD-CP populations, State {}.png".format(g)
    fig_latent_per_state.savefig(os.path.join(pwd, fname))
    plt.close('all')

#%% stats_per_latent_per_state
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    # fig_latent_state_group = plt.figure(figsize=(10, 10))
    # ax1 = fig_latent_state_group.add_subplot(1, 1, 1)
    temp_state_latent = np.zeros((10,10), dtype='float64')
    for i in range(len(videos)):
        v = videos[i]
        state_latent_video_i = Stats_per_latent_per_state[j][i]
        temp = state_latent_video_i[...,0].copy()
        temp[np.isnan(temp)] = 0
        temp_state_latent = temp_state_latent + temp
        fig_latent_state = plt.figure(figsize=(10, 10))
        ax = fig_latent_state.add_subplot(1, 1, 1)
        im = ax.imshow(state_latent_video_i[...,0],vmin=-40, vmax=50)
        ax.set_xlabel('Latent d')
        ax.set_ylabel('State/Motif n')
        ax.set_title('Mean of Latent Vector-{}-{}'.format(titles[j], v))
        plt.colorbar(im)
        plt.show()
        pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'.format()
        os.makedirs(pwd, exist_ok=True)
        fname = "state-latent-{}.png".format(v)
        #fig_latent_state.savefig(os.path.join(pwd, fname))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    im1 = ax.imshow(temp_state_latent/12,vmin=-20, vmax=30)
    ax.set_xlabel('Latent d')
    ax.set_ylabel('State/Motif n')
    ax.set_title('Mean of Latent Vector-{}'.format(titles[j]))
    plt.colorbar(im1)
    plt.show()
    pwd = r'D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\BD20-Jun5-2022\figure\latent_visual'
    os.makedirs(pwd, exist_ok=True)
    fname = "state-latent-{}.png".format(titles[j])
    fig.savefig(os.path.join(pwd, fname))
plt.close('all')