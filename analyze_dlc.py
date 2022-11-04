# Created by Victoria Zhang at 6/5/2022
# File: analyze_dlc.py
# Description: Look at skeleton from dlc, labels from Wendy, and dlc umap
# Scenario:
# Usage:
#%%
import umap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
from vame.analysis.community_analysis import  read_config, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
import tqdm
from pathlib import Path
import torch
from sklearn.cluster import KMeans
from vame.util.auxiliary import read_config
from vame.analysis.pose_segmentation import get_motif_usage, same_parameterization
import pandas as pd
import glob
import datetime
import cv2
#%% Define Project
project_name = 'BD20-Jun5-2022'
path_to_file= 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
start_frame = pd.read_csv('G:\start_frame_vic.csv')
start_frame = start_frame.set_index('video_name').T.to_dict('list')
#%% Read DLC readings
csv_path = r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\videos\pose_estimation'.format(project_name)
confidence = 0.9

for nf, filename in enumerate(os.listdir(csv_path)):
    f_start_frame = start_frame[filename[:-4]][0]

    data = pd.read_csv(os.path.join(path_to_file, 'videos', 'pose_estimation', filename), skiprows=2)
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[f_start_frame:, 1:]
    # get the coordinates for alignment from data table

    for i in range(int(data_mat.shape[1] / 3)):
        temp = data_mat[:, i * 3:(i + 1) * 3]
        for j in temp:
            if j[2] <= confidence:
                j[0], j[1] = np.nan, np.nan
        pose_list = temp[:,0:2] if i == 0 else np.concatenate([pose_list, temp[:, :2]],axis=1)
    pose_list1 = pose_list - np.nan_to_num(pose_list[0,:], nan=0)
    np.save(os.path.join(path_to_file, 'data', 'pose_sequence', filename[:-4] + '-90pct_seq.npy'), pose_list)
    np.save(os.path.join(path_to_file, 'data', 'pose_sequence', filename[:-4] + '-90pct_seq_normalized.npy'), pose_list1)
    time_seq_cat = pose_list if nf == 0 else np.concatenate((time_seq_cat, pose_list),axis=0)
    time_seq_normalize_cat = pose_list1 if nf == 0 else np.concatenate((time_seq_normalize_cat, pose_list1), axis=0)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'cat-90pct_seq.npy'), time_seq_cat)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm-90pct_seq_normalized.npy'), time_seq_normalize_cat)
#%% plot deeplabcut with motif usage
project_name = 'BD20-Jun5-2022'
path_to_file = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}'.format(project_name)
csv_path = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}\videos\pose_estimation'.format(project_name)
confidence = 0.9
n_cluster = 10

load_presaved  = 1
group = ['CP','BD']
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        if not load_presaved:
            f_start_frame = start_frame[v][0]
            label = np.load(r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(project_name, v, n_cluster, n_cluster, v))
            data = pd.read_csv(os.path.join(csv_path, v + '.csv'), skiprows=2)
            frame_count = len(data)
            f_start_frame = frame_count - len(label)  # the old f_start_frame by Victoria

            data_mat = pd.DataFrame.to_numpy(data)
            data_mat = data_mat[f_start_frame:, 1:]
            # get the coordinates for alignment from data table
            for i in range(int(data_mat.shape[1] / 3)):
                temp = data_mat[:, i * 3:(i + 1) * 3]
                for j in temp:
                    if j[2] <= confidence:
                        j[0], j[1] = np.nan, np.nan
                pose_list = temp[:, 0:2] if i == 0 else np.concatenate([pose_list, temp[:, :2]], axis=1)
            np.save(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'), pose_list)
        else:
            label = np.load(
                r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                    project_name, v, n_cluster, n_cluster, v))
            data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))
            d = np.concatenate((label.reshape(-1, 1), data), axis=1)
            df = pd.DataFrame(d, columns=['label', 'l-eyex',	'l-eyey'	,'r-eyex','r-eyey','l_earx','l_eary','r_earx',	'r_eary'	,
                                          'mouthx',	'mouthy','center_neckx','center_necky',	'l_shox','l_shoy','r_shox',	'r_shoy',
                                          'l-elbx',	'l-elby','r-elbx'	,'r-elby'	,'l-handx',	'l-handy',	'r-handx','r-handy',
                                          'center_hipx','center_hipy','l-hipx','l-hipy','r-hipx','r-hipy','l-kneex','l-kneey',
                                          'r-kneex','r-kneey','l-feetx','l-feety','r-feetx','r-feety','center_feetx','center_feety'])

            for body_i in range(1,21):
                x = df.iloc[:,body_i]
                y = df.iloc[:, body_i+1]
                fig, ax = plt.subplots(figsize=(10,5))
                lx = ax.plot(x,label='x',color='b',)
                ly = ax.plot(y,label='y',color='r')
                ax.legend()
                cmap = plt.get_cmap('tab20')
                condition = df['label']
                current_c = condition[0]
                spans = [[0, 0]]
                span_label = [current_c]
                for i, c in enumerate(condition):
                    if c != current_c:
                        # change to i-1 if gap between two conditions is to be left empty
                        spans[-1][-1] = i
                        spans.append([i, None])
                        span_label.append(c)
                        current_c = c
                # assumes that the last condition is not on its own but same as previous
                spans[-1][-1] = len(condition) - 1
                # span_label.append(condition[-1])
                for i in range(len(spans)):
                    span = spans[i]
                    l = span_label[i]
                    ax.axvspan(span[0], span[1], alpha=0.2,color=cmap.colors[int(l* 2 + j)])

                X = np.arange(len(x))
                legend = [Line2D(X, df.iloc[:,body_i], color='b',  label='x'),
                          Line2D(X, df.iloc[:,body_i+1], color='r', label='y'),
                          Patch(facecolor=cmap.colors[int(0 * 2 + j)], label=0),
                          Patch(facecolor=cmap.colors[int(1* 2 + j)], label=1),
                          Patch(facecolor=cmap.colors[int(2 * 2 + j)], label=2),
                          Patch(facecolor=cmap.colors[int(3 * 2 + j)], label=3),
                          Patch(facecolor=cmap.colors[int(4 * 2 + j)], label=4),
                          Patch(facecolor=cmap.colors[int(5 * 2 + j)], label=5),
                          Patch(facecolor=cmap.colors[int(6 * 2 + j)], label=6),
                          Patch(facecolor=cmap.colors[int(7 * 2 + j)], label=7),
                          Patch(facecolor=cmap.colors[int(8 * 2 + j)], label=8),
                          Patch(facecolor=cmap.colors[int(9 * 2 + j)], label=9),
                          ]
                ax.legend(handles=legend)
                plt.title('{}-{}-{}'.format(group[j], v, df.columns[body_i][:-1]))
                plt.show()
                pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\dwell_time_n_dlc'
                fname = "{}-{}_{}_{}_dwell_time_dlc.png".format(group[j],v, n_cluster, df.columns[body_i][:-1])
                fig.savefig(os.path.join(pwd, fname))

#%% Fit Kmeans to cluster 20 temporal sequences

import numpy as np
from sklearn.cluster import KMeans

def kmeans_missing(X, n_clusters, random_state, n_init, max_iter=10):
    """Perform K-Means clustering on data with missing values.

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.

    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            cls = KMeans(n_clusters, init='k-means++', random_state=random_state, n_init=n_init)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return labels, centroids, X_hat

n_cluster = cfg['n_cluster']
random_state = cfg['random_state_kmeans']
n_init = cfg['n_init_kmeans']


labels, centroids, time_seq_hat = kmeans_missing(time_seq_normalize_cat, n_cluster, random_state=random_state, n_init=n_init, max_iter=30)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm_labels.npy'), labels)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm_centroids.npy'), centroids)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm_hat.npy'), time_seq_hat)
#%% Umap embedding of the data
standard_embedding = umap.UMAP(n_components=3,random_state=42).fit_transform(time_seq_hat[:300000,:])
np.save(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\BD20-Jun5-2022\data\time-seq-umap-embed.npy',standard_embedding)
#%%
import plotly.graph_objects as go
#standard_embedding = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\BD20-Jun5-2022\data\time-seq-umap-embed.npy')
ax = plt.axes(projection='3d')
scatter = ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1], standard_embedding[:, 2],
           c=labels[:300000], s=0.1, cmap='Spectral')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Cluster")
ax.add_artist(legend1)
ax.legend()
plt.pause(1)
plt.show()
#%%
ax = plt.axes()
scatter = ax.scatter(standard_embedding[:, 0], standard_embedding[:, 1],
           c=labels[:300000], s=0.1, cmap='Spectral')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Cluster")
ax.add_artist(legend1)
ax.legend()
plt.show()

#%% Read scoring of video
control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
video_path = r'G:\hBPM_Videos'
scoring_path = r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\Scoring\video_score_vz'
bahavior_names =["sit_obj", "sit", "stand_obj", "stand", "walk_obj", "walk", "lie_obj", "lie", "interact", "wear", "exercise"]
total_usage = np.zeros([2,6])
motif_usage_cat = [[],[]]
for nf, filename in enumerate(os.listdir(scoring_path)):

    video_name = filename[:7]
    print(video_name)
    # video = cv2.VideoCapture(filename)
    # duration = video.get(cv2.CAP_PROP_POS_MSEC)
    # frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    data = pd.read_excel(os.path.join(scoring_path, filename), sheet_name='15 template data',skiprows=3, usecols=range(1,12))
    usage = pd.read_excel(os.path.join(scoring_path, filename), sheet_name='15 Data time', skiprows=3,header=None,usecols=range(12, 14))


    usage = usage.dropna()
    usage1 = pd.DataFrame.to_numpy(usage)
    for i in range(len(usage1)):
        t = usage1[i,1]
        minute = t.hour
        second = t.minute
        frame = t.second
        nf = (minute * 60 + second) * 30 + frame
        usage1[i,1] = nf
    if video_name in BD_videos:
        total_usage[0,:] += usage1[:, 1].astype(int)
        motif_usage_cat[1].append(usage1[:, 1].astype(int)/np.sum(usage1[:, 1]))
    if video_name in control_videos:
        total_usage[1,:] += usage1[:, 1].astype(int)
        motif_usage_cat[0].append(usage1[:, 1].astype(int) / np.sum(usage1[:, 1]))

    data = data.dropna(how='all')
    data_mat = pd.DataFrame.to_numpy(data)
    frame_score = []
    for i in range(int(len(data_mat) / 2) -1):
        start = data_mat[i*2, :]
        end = data_mat[i*2 +1, :]
        df1 = np.asarray([start, end])
        df2 = df1.copy()
        for j, row in enumerate(df1):
            for k, t in enumerate(row):
                if type(t) is datetime.time:
                    minute = t.hour
                    second = t.minute
                    frame = t.second
                    nf = (minute * 60 + second) * 30 + frame
                    df2[j][k] = nf
        start_frames = df2[0,:].reshape(1,-1) if i == 0 else np.concatenate((start_frames, df2[0,:].reshape(1,-1)), axis=0)
        end_frames = df2[1, :].reshape(1,-1) if i == 0 else np.concatenate((end_frames, df2[1, :].reshape(1,-1)), axis=0)

    start_frames_df = pd.DataFrame(start_frames)
    start_frames_df.columns = bahavior_names
    end_frames_df = pd.DataFrame(end_frames)
    end_frames_df.columns = bahavior_names
    output_csv = os.path.join(scoring_path, filename[:8] + 'frames.xlsx')
    # with pd.ExcelWriter(output_csv) as writer:
    #     start_frames_df.to_excel(writer, sheet_name='start')
    #     end_frames_df.to_excel(writer, sheet_name='end')
#%%
import matplotlib.pyplot as plt
titles = ["BD","CP"]
fig, axs = plt.subplots(1, 2, figsize=(20,10))
for j, videos in enumerate([control_videos, BD_videos]):
    # Plot histogram

    axs[j].bar(np.arange(0, 6 , 1), list(total_usage[j,:]/np.sum(total_usage[j,:])),
            width = 0.4)
    axs[j].set_title(titles[j])
    axs[j].set_xlabel("motifs")
    axs[j].set_ylabel("occurrence (%)")
    axs[j].set_xticks(np.arange(0, 6 , 1))
    axs[j].set_xticklabels(usage1[:, 0])
plt.show()
#%%
motif_usage_cat = np.asarray(motif_usage_cat)
from scipy import stats
for i in range(30):
    CP = motif_usage_cat[0,:,i].reshape(-1,1)
    BD = motif_usage_cat[1,:,i].reshape(-1,1)
    s = stats.ttest_rel(CP, BD)
    print("motif {}:{}, {}".format(i,s.statistic, s.pvalue))

#%% Motif Usage



files = []
for file in cfg['video_sets']:
    files.append(file)
motif_usages = []
cluster_centers = []
cluster_labels = []
idx = 0
for i, file in enumerate(files):
    pose_list = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', filename[:-4] + '-90pct_seq.npy'))
    file_len = pose_list.shape[0]
    cluster_labels.append(labels[idx:idx + file_len])
    cluster_centers.append(centroids)

    motif_usage = get_motif_usage(labels[idx:idx + file_len], n_cluster)
    motif_usages.append(motif_usage)
    idx += file_len
