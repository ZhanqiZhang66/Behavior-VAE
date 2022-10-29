# Created by Victoria Zhang at 8/31/2022
# File: sliding_window.py
# Description: 
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
import time
import math
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
from sklearn.decomposition import PCA
#%%
load_precomputed_sliding_window = True
#%%
#%%
if not load_precomputed_sliding_window:
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
    def minimum_state_duration(labels, n_cluster):
        return None

    def calculate_distance(starting_x, starting_y, destination_x, destination_y):
        distance = math.hypot(destination_x - starting_x,
                              destination_y - starting_y)  # calculates Euclidean distance (straight-line) distance between two points
        return distance
    def compute_velocity(pose, window_size):

        velocity = []
        for i in range(int(pose.shape[1] / 3)):
            temp = data_mat[:, i * 3:(i + 1) * 3]

            x_s = j[0]
            y_s = j[1]
            dist_traveled = 0
            for (x,y) in (x_s, y_s):
                dist_seg = calculate_distance(x, y)
                dist_traveled += dist_seg
            velocity.append(dist_traveled/window_size)
        return velocity
    #%% Sliding window of 5 min analysis

    project_name = 'BD20-Jun5-2022'
    config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
    cfg = read_config(config)
    cluster_start = cfg['time_window'] / 2
    n_cluster = 10
    d_latent = 10
    model_name = 'VAME'
    control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
    BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
    start_frame = pd.read_csv('D:\OneDrive - UC San Diego\GitHub\Behavior-VAE\start_frame_vic.csv')

    titles = ["CP", "BD"]

    slide_window = {
      "subject": [],
      "start_frame": [],
      "is_BD": [],
      "entropy": [],
      "num_zero_row":[],
      "num_one_item": [],
      "num_zero_item":[],
      "motif0_usage_freq": [],
      "motif1_usage_freq": [],
      "motif2_usage_freq": [],
      "motif3_usage_freq": [],
      "motif4_usage_freq": [],
      "motif5_usage_freq": [],
      "motif6_usage_freq": [],
      "motif7_usage_freq": [],
      "motif8_usage_freq": [],
      "motif9_usage_freq": [],
      "latent_volume_all_motifs": [],
      "latent_volume_motif0": [],
      "latent_volume_motif1": [],
      "latent_volume_motif2": [],
      "latent_volume_motif3": [],
      "latent_volume_motif4": [],
      "latent_volume_motif5": [],
      "latent_volume_motif6": [],
      "latent_volume_motif7": [],
      "latent_volume_motif8": [],
      "latent_volume_motif9": [],
    }

    csv_path = r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\videos\pose_estimation'.format(project_name)


    #%%

    pca = PCA(n_components=3)
    for j, videos in enumerate([control_videos, BD_videos]):
        for i in range(len(videos)):
            start = time.time()
            v = videos[i]
            print("Loading {} data...".format(v))
            folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
            label = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(project_name, v,n_cluster,n_cluster,v))
            latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 30

            v_index = start_frame.loc[start_frame['video_name'] == v].index.values[0]
            door_close_time = int(start_frame.loc[v_index, 'door_close'])
            start_time = start_frame.loc[v_index, 'n']
            window_size = int(3 * 60 * 30)
            offset = int(door_close_time - start_time)



            data = pd.read_csv(os.path.join(csv_path, v +'.csv'), skiprows=2)
            data_mat = pd.DataFrame.to_numpy(data)
            data_mat = data_mat[start_time:, 1:]

            # get the coordinates for alignment from data table

            for k in range(len(label) - window_size + 1):
                if k % 1000 == 0:
                    print(" Sliding window {} - {}...".format(offset + k, window_size + offset + k))
                window_label = label[offset + k: window_size + offset + k]
                window_motif_usage = get_motif_usage(window_label, n_cluster)
                window_latent_vector = latent_vector[offset + k: window_size + offset + k]
                window_transition_matrix = compute_transition_matrices([v], [window_label], n_cluster)
                num_zero_row, num_one_item, num_zero_item = count_zeros(window_transition_matrix[0])
                entropy = compute_l0_entropy(window_transition_matrix[0], window_label[-1])
                velocity = compute_velocity(data_mat[offset + k: window_size + offset + k], window_size)

                slide_window["subject"].append(v)
                slide_window["start_frame"].append(k)
                slide_window["is_BD"].append(j)
                slide_window["entropy"].append(entropy)
                slide_window["num_zero_row"].append(num_zero_row)
                slide_window["num_one_item"].append(num_one_item)
                slide_window["num_zero_item"].append(num_zero_item)
                for i in range(n_cluster):
                    slide_window['motif{}_usage_freq'.format(i)].append(window_motif_usage[i]/np.sum(window_motif_usage))

                slide_window["motif_usage_freq"].append(window_motif_usage/np.sum(window_motif_usage))

                K = np.cov(window_latent_vector.T)
                volume_of_all = np.trace(K)
                slide_window["latent_volume_all_motifs"].append(volume_of_all)

                latent_volume_per_motif = []
                for g in range(n_cluster):
                    i = np.where(window_label == g)
                    if len(i[0]):
                        latent_sub_g = window_latent_vector[i]
                        K_sub = np.cov(latent_sub_g.T)
                        volume_of_group_sub = np.trace(K_sub)
                    else:
                        volume_of_group_sub = 0
                    latent_volume_per_motif.append(volume_of_group_sub)
                    slide_window['latent_volume_motif{}'.format(g)].append(volume_of_group_sub)

                slide_window["latent_volume_per_motif"].append(latent_volume_per_motif)
            end = time.time()
            print(f"Runtime of one video is {end - start}")
    #%%
    pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\data\slide_window2.csv'
    ds_new = pd.DataFrame.from_dict(slide_window)
    ds_new.to_csv(pwd)
#%%
if load_precomputed_sliding_window:
    pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\data\slide_window2.csv'
    ds = pd.read_csv(pwd)


    project_name = 'BD20-Jun5-2022'
    n_cluster = 10
    d_latent = 10
    model_name = 'VAME'
    control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
    BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
    start_frame = pd.read_csv('G:\start_frame.csv')
    diagnosis_score = pd.read_csv(
        'D:\OneDrive - UC San Diego\Bahavior_VAE_data\Participant_videos_attributes\First-24-Videos\Subject_24ID-BDs-HCs.csv',
        encoding='windows-1252')
    YMRS = diagnosis_score[['Subject ID', 'YMRS (max score, 60. Pts are ineligible > 12)']]
    YMRS = YMRS.set_index('Subject ID').T.to_dict('list')
    HAM_D = diagnosis_score[['Subject ID', 'HAM-D']]
    HAM_D = HAM_D.set_index('Subject ID').T.to_dict('list')
#%% plot average metric per population
num_metrics = 5
metric_names = ["entropy",
                "num_zero_row",
                "num_one_item",
                "num_zero_item",
                "latent_volume_all_motifs" ,
                "start_frame",
                "is_BD"
                ]

lims = [[0, 2.3], [-5, 15], [-4, 8], [30, 120]]

sns.set_style('darkgrid')
CP_idx = np.zeros(12)
BD_idx = np.ones(12)
sns.set_style('darkgrid')
for i in range(num_metrics):
    fig, axes = plt.subplots(1, figsize=(10, 5))
    t_max = ds["start_frame"].max()
    for group in range(2):
        ds1 = ds[ds["is_BD"] == group]
        metric_mean_over_sub = []
        for t in range(t_max):
            ds_t = ds1[ds1["start_frame"] == t]
            metric_mean_over_sub.append(ds_t[metric_names[i]].mean())
        x = np.arange(t_max)
        line = axes.plot(x, metric_mean_over_sub, color='C{}'.format(group))

        axes.set_title('average {}'.format(metric_names[i]))
        # axes[i].set_ylim(lims[i])
        axes.set_xlabel('population')
    plt.show()

#plt.suptitle("sliding window")

#%% Plot per patient L0 metrics
num_metrics = 1
CP_idx = np.zeros(12)
BD_idx = np.ones(12)

for i in range(num_metrics):
    for j, videos in enumerate([control_videos, BD_videos]):
        for sub in range(12):
            fig, axes = plt.subplots(1, figsize=(10, 5))
            sns.set_style('darkgrid')
            if sub == 1 or sub == 12:
                leg = True
            else:
                leg = False
            sub_name = videos[sub]
            ds1 = ds[ds["subject"] == sub_name]
            line = sns.scatterplot(data=ds1, x="start_frame", y=metric_names[i],
                                hue="is_BD", palette={0:'C0', 1:'C1'}, ax=axes, legend=leg,
                                linewidth=0, alpha = 0.2, s=2)
            axes.set_ylim(lims[i])
            axes.set_title('subject {}'.format(sub_name))
            axes.set_xlabel('population')
            #axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.show()
            pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\transtion_metrics'
            fname = "{}_{}.png".format(metric_names[i], sub_name)
            fig.savefig(os.path.join(pwd, fname))

#%% First half second half analysis



#%% Per patient, entropy, and latent volume per motif second half/ first half

groups = ['CP', 'BD']
entropy = []
latent_volume0 = []
latent_volume1 = []
latent_volume2 = []
latent_volume3 = []
latent_volume4 = []
latent_volume5 = []
latent_volume6 = []
latent_volume7 = []
latent_volume8 = []
latent_volume9 = []
is_BD = []
latent_d = []
for j, videos in enumerate([control_videos, BD_videos]):
    for sub in range(12):
        sub_name = videos[sub]
        df1 = ds[ds["subject"] == sub_name]
        total_len = len(df1)
        df2 = df1["entropy"]
        entropy_first_half = np.nanmean(df2[:total_len//2])
        entropy_second_half = np.nanmean(df2[total_len //2:])

        entropy.append(entropy_second_half-entropy_first_half)
        is_BD.append(j)
        for d in range(n_cluster):
            y = df1['latent_volume_motif{}'.format(d)].to_numpy()
            latent_v_first_half = np.nanmean(y[:total_len // 2])
            latent_v_second_half = np.nanmean(y[total_len // 2:])
            eval('latent_volume{}'.format(d)).append(latent_v_second_half - latent_v_first_half)
entropy_df = pd.DataFrame(np.asarray([entropy, is_BD]).T, columns=['metric','is_BD'])



fig, ax = plt.subplots(1, figsize=(10, 5))
sns.boxplot(y="metric", x='is_BD', hue='is_BD', data=entropy_df, orient="v")
ax.set_xticklabels(['CP','BD'])
ax.set_title('change in entropy')
ax.set_ylabel('second half - first half ')
fig.show()
pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\transtion_metrics'
fname = "entropy_first_over_second.png"
fig.savefig(os.path.join(pwd, fname))

latent_df = pd.DataFrame(np.asarray([latent_volume0, latent_volume1, latent_volume2, latent_volume3, latent_volume4,
                                     latent_volume5, latent_volume6, latent_volume7, latent_volume8, latent_volume9,
                                     is_BD]).T,
                         columns=['latent_volume0', 'latent_volume1', 'latent_volume2', 'latent_volume3', 'latent_volume4',
                                  'latent_volume5', 'latent_volume6', 'latent_volume7', 'latent_volume8', 'latent_volume9','is_BD'])

for d in range(n_cluster):
    fig, ax = plt.subplots(1, figsize=(10, 5))
    sns.boxplot(y="latent_volume{}".format(d), x='is_BD', hue='is_BD', data=latent_df, orient="v")
    ax.set_xticklabels(['CP', 'BD'])
    ax.set_ylabel('second half  - first half ')
    ax.set_title('change in latent volume motif {}'.format(d))
    ax.set_ylim([-900, 600])
    fig.show()
    pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\latent_slide_window'
    fname = "latent_volume_motif_{}_first_over_second.png".format(d)
    fig.savefig(os.path.join(pwd, fname))
#%%
from scipy import stats
CP = entropy_df['metric'][:12].to_numpy()
BD = entropy_df['metric'][12:].to_numpy()
res = stats.ttest_ind(CP, BD)
f= stats.f_oneway(CP, BD)

F = np.var(CP) / np.var(BD)
df1 = len(CP) - 1
df2 = len(BD) - 1
alpha = 0.05 #Or whatever you want your alpha to be.
p_value = stats.f.cdf(F, df1, df2)


#%% Plot per patient change of dwell time, and latent volume
from itertools import zip_longest
latent_d = 10
n_cluster= 10
CP_idx = np.zeros(12)
BD_idx = np.ones(12)
cmap = plt.get_cmap('tab20')
lims = [[-500, 2000],[-0.2, 1.2]]
groups = ['Control', 'BD']
CP_mean_motifs = [[],[]]
BD_mean_motifs = [[],[]]
for j, videos in enumerate([control_videos, BD_videos]):
    group = groups[j]
    mean_motif_freq = [[] for _ in range(n_cluster)]
    mean_motif_volume = [[] for _ in range(n_cluster)]
    for sub in range(12):
        fig, ax = plt.subplots(1, figsize=(10, 5))
        fig1, ax1 = plt.subplots(1, figsize=(10, 5))
        sub_name = videos[sub]
        ds1 = ds[ds["subject"] == sub_name]
        for d in range(n_cluster):
            x = ds1["start_frame"].to_numpy()
            y = ds1['latent_volume_motif{}'.format(d)].to_numpy()
            z = ds1['motif{}_usage_freq'.format(d)].to_numpy()
            mean_motif_freq[d].append(z)
            mean_motif_volume[d].append(y)
            ax.scatter(x, y, norm=plt.Normalize(vmin=0, vmax=9), color=cmap(d * 2 + j), s=2, label='%d' % d)
            ax1.scatter(x, z, norm=plt.Normalize(vmin=0, vmax=9), color=cmap(d * 2 + j), s=2, label='%d' % d)
        ax.set_ylim(lims[0])
        ax.set_title('{}-{}-latent volume per motif'.format(group, sub_name))
        ax.set_xlabel('time (frames)')
        ax1.set_ylim(lims[1])
        ax1.set_title('{}-{}-motif frequency'.format(group, sub_name))
        ax1.set_xlabel('time (frames)')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\latent_slide_window'
        fname = "{}_{}.png".format('latent_colume', sub_name)
        #fig.savefig(os.path.join(pwd, fname))

        pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\motif_freq_slide_window'
        fname = "{}_{}.png".format('motif_usage', sub_name)
        #fig1.savefig(os.path.join(pwd, fname))
    if j == 0:
        CP_mean_motifs = [mean_motif_freq, mean_motif_volume]
    if j == 1:
        BD_mean_motifs = [mean_motif_freq, mean_motif_volume]
#%% Plot per population change of dwell time, and latent volume
def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


latent_d = 10
n_cluster= 10
CP_idx = np.zeros(12)
BD_idx = np.ones(12)
cmap = plt.get_cmap('tab20')
lims = [[-500, 2000],[-0.2, 1.2]]
groups = ['CP', 'BD']


for d in range(n_cluster):
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig1, ax1 = plt.subplots(1, figsize=(10, 5))
    for j, videos in enumerate([control_videos, BD_videos]):
        group = groups[j]
        mean_motif_freq = CP_mean_motifs[0][d] if j == 0 else BD_mean_motifs[0][d]
        z, error_z = tolerant_mean(mean_motif_freq)


        mean_motif_volume = CP_mean_motifs[1][d] if j == 0 else BD_mean_motifs[1][d]
        y, error_y = tolerant_mean(mean_motif_volume)

        x = np.arange(len(y)) + 1
        ax.fill_between(x, y - error_y, y + error_y, norm=plt.Normalize(vmin=0, vmax=9),
                        alpha=0.1, facecolor=cmap(d * 2 + j))
        ax.plot(x, y, color=cmap(d * 2 + j), label='{}-{}'.format(group, d))

        x = np.arange(len(z)) + 1
        ax1.fill_between(x, z - error_z, z + error_z, norm=plt.Normalize(vmin=0, vmax=9),
                         alpha=0.1, facecolor=cmap(d * 2 + j))
        ax1.plot(x, z, color=cmap(d * 2 + j), label='{}-{}'.format(group, d))

    ax.axhline(0, color='white',linestyle="dashed")
    ax.set_ylim(lims[0])
    ax.set_title('{}-{}-latent volume motif {}'.format('BD-CP', 'average', d))
    ax.set_xlabel('time (frames)')

    ax1.axhline(0, color='white',linestyle="dashed")
    ax1.set_ylim(lims[1])
    ax1.set_title('{}-{}-motif{} frequency'.format('BD-CP', 'average', d))
    ax1.set_xlabel('time (frames)')
    ax.legend()
    ax1.legend()
    # plt.show()
    pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\latent_slide_window'
    fname = "{}_{}_motif{}.png".format('latent_volume', 'BD-CP', d)
    #fig.savefig(os.path.join(pwd, fname))

    pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\motif_freq_slide_window'
    fname = "{}_{}_motif{}.png".format('motif_usage', 'BD-CP', d)
    #fig1.savefig(os.path.join(pwd, fname))
#%%
# num_metrics = 5
# CP_idx = np.zeros(12)
# BD_idx = np.ones(12)
#
#
# for i in range(num_metrics):
#     fig, axes = plt.subplots(1, figsize=(10, 5))
#     sns.set_style('darkgrid')
#     for j, videos in enumerate([control_videos, BD_videos]):
#         for sub in range(12):
#             if sub == 1 or sub == 12:
#                 leg = True
#             else:
#                 leg = False
#             sub_name = videos[sub]
#             ds1 = ds[ds["subject"] == sub_name]
#             line = sns.stripplot(data=ds1, x="start_frame", y=metric_names[i],
#                                 hue="is_BD", palette={0:'C0', 1:'C1'}, ax=axes, legend=leg,
#                                 linewidth=0, alpha = 0.2)
#     # axes[i].set_ylim(lims[i])
#     axes.set_xlabel('population')
#     axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#     plt.show()