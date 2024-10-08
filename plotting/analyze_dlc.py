# Created by Victoria Zhang at 6/5/2022
# File: analyze_dlc.py
# Description: Look at skeleton from dlc, labels from Wendy, and dlc umap
# Scenario: We want to know if DLC labels can be used to cluster behaviors
# Usage:
#%%
import umap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pylab as pl
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
from pathlib import Path
import pathlib
from data.load_data import load_pt_data
from analysis.Classifiers.Generation.utils import my_colormap
from plotting.get_paths import get_my_path
#%%
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']
#%% Flags
# import argparse
#
# parser = argparse.ArgumentParser(description='Load DLC pose estimations, run stat tests, and show result')
# parser.add_argument('--load_DLC_filtered', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
#
# args = parser.parse_args()
# print(args.accumulate(args.integers))


#%% plot deeplabcut with motif usage, and deeplabcut trajectory in room
load_filtered_DLC = 1 # 0: remove markers under 90% confidence, otherwise, 1: just loaded the filtered DLC
compute_significance = 1
# Fig. 1c
show_DLC_in_room_over_time = 1 # show DLC trajectory in room

#%% Define Project
project_name = 'BD25-HC25-final-May17-2023'
project_path = f'{onedrive_path}\Behavior_VAE_data\{project_name}'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(project_path,"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
model_name = 'VAME'
path_to_file = cfg['project_path']

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']

dlc_labels = ['label', 'l-eyex','l-eyey','r-eyex','r-eyey','l_earx','l_eary','r_earx',	'r_eary',
              'mouthx',	'mouthy','center_neckx','center_necky',	'l_shox','l_shoy','r_shox',	'r_shoy',
              'l-elbx',	'l-elby','r-elbx'	,'r-elby'	,'l-handx',	'l-handy',	'r-handx','r-handy',
              'center_hipx','center_hipy','l-hipx','l-hipy','r-hipx','r-hipy','l-kneex','l-kneey',
              'r-kneex','r-kneey','l-feetx','l-feety','r-feetx','r-feety','center_feetx','center_feety'
              ]
n_subject_in_population = len(control_videos)

csv_path = os.path.join(cfg['project_path'],"videos","pose_estimation")
confidence = 0.9
group = ['CP','BD']
temp_win = cfg['time_window']
#%% Read DLC readings
if not load_filtered_DLC:
    for nf, filename in enumerate(os.listdir(csv_path)):
        v = filename[:-4]
        v_index = start_frame.loc[start_frame['video_name'] == v].index.values[0]
        f_start_frame = start_frame.loc[v_index, 'door_close']


        data = pd.read_csv(os.path.join(path_to_file, 'videos', 'pose_estimation', filename), skiprows=2)
        data_mat0 = pd.DataFrame.to_numpy(data)
        data_mat = data_mat0[f_start_frame:, 1:]
        data_mat_ori = data_mat0[:, 1:]
        # get the coordinates for alignment from data table

        for i in range(int(data_mat.shape[1] / 3)):
            temp = data_mat[:, i * 3:(i + 1) * 3]
            temp_org = data_mat_ori[:, i * 3:(i + 1) * 3]
            for j in temp:
                if j[2] <= confidence:
                    j[0], j[1] = np.nan, np.nan
            pose_list = temp[:,0:2] if i == 0 else np.concatenate([pose_list, temp[:, :2]],axis=1)
            for idx, k in enumerate(temp_org):
                lower_bound = min(0, idx - 60)
                upper_bound = min(len(data_mat_ori), idx + 60)
                if k[2] <= confidence:
                    k[0], k[1] = np.nanmean(temp_org[lower_bound:upper_bound, 0]), np.nanmean(temp_org[lower_bound:upper_bound, 1])
            pose_list_org = temp_org[:,0:2] if i == 0 else np.concatenate([pose_list_org, temp_org[:, :2]],axis=1)
        print(f"Saving %{confidence*100} {v} pose estimation data...\n")
        pose_list1 = pose_list - np.nan_to_num(pose_list[0,:], nan=0)
        #np.save(os.path.join(path_to_file, 'data', 'pose_sequence', filename[:-4] + '-90pct_seq.npy'), pose_list)
        np.save(os.path.join(path_to_file, 'data', 'pose_sequence', filename[:-4] + '-90pct_seq_original_smoothed.npy'), pose_list_org)
        # np.save(os.path.join(path_to_file, 'data', 'pose_sequence', filename[:-4] + '-90pct_seq_normalized.npy'), pose_list1)
        time_seq_cat = pose_list if nf == 0 else np.concatenate((time_seq_cat, pose_list),axis=0)
        # time_seq_normalize_cat = pose_list1 if nf == 0 else np.concatenate((time_seq_normalize_cat, pose_list1), axis=0)
    # np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'cat-90pct_seq.npy'), time_seq_cat)
# np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm-90pct_seq_normalized.npy'), time_seq_normalize_cat)


#%%

#%% filter DLC keypoints by confidence level
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]

        f_start_frame = start_frame[v][0]  # f_start_frame = frame_count - len(label)  # the old f_start_frame by Victoria

        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name,
                                                                                           v, n_cluster, n_cluster, v))
        data = pd.read_csv(os.path.join(csv_path, v + '.csv'), skiprows=2)
        frame_count = len(data)

        data_mat_original = pd.DataFrame.to_numpy(data)
        data_mat = data_mat_original[f_start_frame:, 1:]
        # get the coordinates for alignment from data table
        for i in range(int(data_mat.shape[1] / 3)):
            temp = data_mat[:, i * 3:(i + 1) * 3]
            temp_original = data_mat_original[:, i * 3:(i + 1) * 3]

            # replace markers < 90 confidence to be nanmean average in 30 seconds period
            for j_idx, j in enumerate(temp):
                if j[2] <= confidence:
                    start = np.max(j_idx-450, 0)
                    j[0], j[1] = np.nanmean(temp[start:j_idx+450,0], 0), np.nanmean(temp[start:j_idx+450, 1], 0)
                    temp[j_idx, 0] = j[0]
                    temp[j_idx, 1] = j[1]
            pose_list = temp[:, 0:2] if i == 0 else np.concatenate([pose_list, temp[:, :2]], axis=1)
            for k_idx, k in enumerate(temp_original):
                if k[2] <= confidence:
                    start = np.max(k_idx-450, 0)
                    k[0], k[1] = np.nanmean(temp_original[start:k_idx+450,0], 0), np.nanmean(temp_original[start:k_idx+450, 1], 0)
                    temp_original[k_idx, 0] = k[0]
                    temp_original[k_idx, 1] = k[1]
            pose_list_original = temp_original[:, 0:2] if i == 0 else np.concatenate([pose_list_original, temp_original[:, :2]], axis=1)
        pwd = r'{}\data\pose_sequence'.format(path_to_file)
        Path(pwd).mkdir(exist_ok=True)
        np.save(os.path.join(pwd, v + '-90pct_seq.npy'), pose_list)
        np.save(os.path.join(pwd, v + '-90pct_seq_original.npy'), pose_list_original)
print("Finished cleaning or loading the DLC data\n")
#%%  Load filtered DLC keypoints
unfiltered_DLC_data_cat = []
data_length = []
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]

        f_start_frame = start_frame[v][0]  # f_start_frame = frame_count - len(label)

        label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path,
                                                                                           project_name,
                                                                                           v, n_cluster,
                                                                                           n_cluster, v))
        data = pd.read_csv(os.path.join(csv_path, v + '.csv'), skiprows=2)
        frame_count = len(data)

        data_mat_original = pd.DataFrame.to_numpy(data)
        data_mat = data_mat_original[f_start_frame:, 1:]

        unfiltered_DLC_data = pd.read_csv(os.path.join(path_to_file, 'videos', 'pose_estimation', v+'.csv'), skiprows=2)
        unfiltered_DLC_data_np = pd.DataFrame.to_numpy(unfiltered_DLC_data)
        unfiltered_DLC_data_np = unfiltered_DLC_data_np[f_start_frame:, 1:]

        save_data = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        if not os.path.exists(
                os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")):
            try:
                pathlib.Path(
                    os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster),
                                 "")).mkdir(parents=True)

                print('create path ----')
            except OSError as error:
                print(error)

        print("loading ", v)
        #np.save(os.path.join(save_data, 'dlc_vector_' + v), unfiltered_DLC_data_np)
        unfiltered_DLC_data_cat.append(unfiltered_DLC_data_np)
        data_length.append(len(unfiltered_DLC_data_np))
unfiltered_DLC_data_cat = np.concatenate(unfiltered_DLC_data_cat, axis=0)
#%%  Compute K-means clustering on raw DLC data
print("Compute K-means clustering on raw DLC data\n")
kmeans = KMeans(n_clusters=n_cluster, random_state=0, n_init=10).fit(unfiltered_DLC_data_cat)
kmeans_labels = kmeans.labels_
label = kmeans.predict(unfiltered_DLC_data_cat)
clust_center = kmeans.cluster_centers_
idx = 0
count = 0
labels = []
cluster_centers = []
motif_usages = []
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("K-means clustering (same parameterization) for {}...".format(v))

        file_len = data_length[count]
        labels.append(label[idx:idx + file_len])
        cluster_centers.append(clust_center)
        motif_usage = get_motif_usage(label[idx:idx + file_len], n_cluster)
        motif_usages.append(motif_usage)
        np.save(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v), label[idx:idx + file_len])
        np.save(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_cluster_center_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v), clust_center)
        np.save(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_motif_usage_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v), motif_usage)

        idx += file_len
        count += 1
print("\n Finished K-means clustering on raw DLC data")
#%% Fit Kmeans to cluster DLC temporal sequences
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


labels, centroids, time_seq_hat = kmeans_missing(time_seq_normalize_cat, n_cluster, n_init=n_init, max_iter=30)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm_labels.npy'), labels)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm_centroids.npy'), centroids)
np.save(os.path.join(path_to_file, 'data', 'pose_sequence', 'catnorm_hat.npy'), time_seq_hat)
#%%  Compute K-means clustering on filtered DLC data
import random
print("Compute K-means clustering on filtered DLC data\n")
n_cluster = cfg['n_cluster']
random_state = cfg['random_state_kmeans']
n_init = cfg['n_init_kmeans']
random.seed(10)
label, clust_center, time_seq_hat = kmeans_missing(unfiltered_DLC_data_cat, n_cluster, random_state=None, n_init=n_init, max_iter=30)

idx = 0
count = 0
labels = []
cluster_centers = []
motif_usages = []
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("K-means clustering (same parameterization) for {}...".format(v))

        file_len = data_length[count]
        labels.append(label[idx:idx + file_len])
        cluster_centers.append(clust_center)
        motif_usage = get_motif_usage(label[idx:idx + file_len], n_cluster)
        motif_usages.append(motif_usage)
        np.save(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\filtered_DLC_{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v), label[idx:idx + file_len])
        np.save(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\filtered_DLC_{}_cluster_center_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v), clust_center)
        np.save(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\filtered_DLC_{}_motif_usage_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v), motif_usage)

        idx += file_len
        count += 1
print("\n Finished K-means clustering on filtered DLC data")
# %% ===================PLOT====================

#%% Umap embedding of the data
standard_embedding = umap.UMAP(n_components=3,random_state=42).fit_transform(time_seq_hat[:300000,:])
np.save(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\BD20-Jun5-2022\data\time-seq-umap-embed.npy',standard_embedding)
#%% Plot keypoints overlapping motif segmentations [three approaches]
from analysis.Classifiers.Generation.motif_usage import load_motif_labels
sLabelPath  = r'{}\S3D\s3d_labels\s3d_labels_{}.npy'.format(data_path, "{}", "{}")
mLabelPath  = r'{}\MMAction\mmaction_labels\mmaction_labels_{}.npy'.format(data_path, "{}", "{}")
sLabels = load_motif_labels(sLabelPath, videos, 27000)
mLabels = load_motif_labels(mLabelPath, videos, 27000)
titles = ["vame", 'dlc k-means', 'scores', 'scored_merged']
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print(v)

        data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))

        dlc_cluster_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\filtered_DLC_{}_km_label_{}.npy'.format(onedrive_path,
                                                                                               project_name, v,
                                                                                               n_cluster, n_cluster,
                                                                                               v))

        score_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\score_labels_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster,  v))


        score_label[score_label<0] = -1
        # bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
        score_label_merged = score_label.copy()
        score_label_merged[score_label_merged == 1] = 0
        score_label_merged[score_label_merged == 3] = 2
        score_label_merged[score_label_merged == 5] = 4
        score_label_merged[score_label_merged == 7] = 6
        # latent vector has time window of temp_win, thus is temp_win shorter than data
        # we need to crop data to match latent vector
        labels = [vame_label,
                  dlc_cluster_label[temp_win // 2:-temp_win // 2],
                  score_label,
                  score_label_merged,
                  ]

        # a colored line plot for every body keypoint
        for body_i in range(39, 41, 2):
            fig, axs = plt.subplots(6, 1, figsize=(12, 12))
            # plot motif segmentation in three approaches [vame, dlc, score]
            for fig_i in range(len(labels)):
                d = np.concatenate((labels[fig_i].reshape(-1, 1), data[temp_win // 2:-temp_win // 2]), axis=1)

                df = pd.DataFrame(d, columns=dlc_labels)

                cmap = plt.get_cmap('tab20')
                cividis_cmap = plt.get_cmap('cividis')
                uniform_colors = cividis_cmap(np.linspace(0, 1, 20))
                condition = df['label']
                current_c = condition[0]
                spans = [[0, 0]]
                span_label = [current_c]
                for ii, c in enumerate(condition):
                    if c != current_c:
                        # change to i-1 if gap between two conditions is to be left empty
                        spans[-1][-1] = ii
                        spans.append([ii, None])
                        span_label.append(c)
                        current_c = c
                # assumes that the last condition is not on its own but same as previous
                spans[-1][-1] = len(condition) - 1
                # span_label.append(condition[-1])
                for iii in range(len(spans)):
                    span = spans[iii]
                    l_label = span_label[iii]
                    axs[fig_i].axvspan(span[0], span[1], alpha=0.2, color=cmap.colors[int(l_label * 2 + j)])

                X = np.arange(len(x))
                if fig_i == 0 or fig_i == 1:
                    spans[-1][-1] = len(condition) - 1
                    # span_label.append(condition[-1])
                    for iii in range(len(spans)):
                        span = spans[iii]
                        l_label = span_label[iii]
                        axs[fig_i].axvspan(span[0], span[1], alpha=1, color=cmap.colors[int(l_label * 2 + j)])

                    legend = [Line2D(X, df.iloc[:, body_i], color='k', label='x'),
                              Patch(facecolor=cmap.colors[int(0 * 2 + j)], alpha=1, label=0),
                              Patch(facecolor=cmap.colors[int(1 * 2 + j)], alpha=1, label=1),
                              Patch(facecolor=cmap.colors[int(2 * 2 + j)], alpha=1, label=2),
                              Patch(facecolor=cmap.colors[int(3 * 2 + j)], alpha=1, label=3),
                              Patch(facecolor=cmap.colors[int(4 * 2 + j)], alpha=1, label=4),
                              Patch(facecolor=cmap.colors[int(5 * 2 + j)], alpha=1, label=5),
                              Patch(facecolor=cmap.colors[int(6 * 2 + j)], alpha=1, label=6),
                              Patch(facecolor=cmap.colors[int(7 * 2 + j)], alpha=1, label=7),
                              Patch(facecolor=cmap.colors[int(8 * 2 + j)], alpha=1, label=8),
                              Patch(facecolor=cmap.colors[int(9 * 2 + j)], alpha=1, label=9),
                              ]
                elif fig_i == 2:
                    spans[-1][-1] = len(condition) - 1
                    # span_label.append(condition[-1])
                    for iii in range(len(spans)):
                        span = spans[iii]
                        l_label = span_label[iii]
                        axs[fig_i].axvspan(span[0], span[1], alpha=1, color=uniform_colors[int(l_label * 2 + j)])

                    legend = [Line2D(X, df.iloc[:, body_i], color='k', label='x'),
                              Patch(facecolor=uniform_colors[int(0 * 2 + j)], alpha=0.8, label="sit"),
                              Patch(facecolor=uniform_colors[int(1 * 2 + j)], alpha=0.8, label="sit_obj"),
                              Patch(facecolor=uniform_colors[int(2 * 2 + j)], alpha=0.8, label="stand"),
                              Patch(facecolor=uniform_colors[int(3 * 2 + j)], alpha=0.8, label="stand-obj"),
                              Patch(facecolor=uniform_colors[int(4 * 2 + j)], alpha=0.8, label="walk"),
                              Patch(facecolor=uniform_colors[int(5 * 2 + j)], alpha=0.8, label="walk_obj"),
                              Patch(facecolor=uniform_colors[int(6 * 2 + j)], alpha=0.8, label="lie"),
                              Patch(facecolor=uniform_colors[int(7 * 2 + j)], alpha=0.8, label="lie_obj"),
                              Patch(facecolor=uniform_colors[int(8 * 2 + j)], alpha=0.8, label="interact"),
                              Patch(facecolor=uniform_colors[int(9 * 2 + j)], alpha=0.8, label="wear"),
                              ]
                else:
                    spans[-1][-1] = len(condition) - 1
                    # span_label.append(condition[-1])
                    for iii in range(len(spans)):
                        span = spans[iii]
                        l_label = span_label[iii]
                        axs[fig_i].axvspan(span[0], span[1], alpha=1, color=uniform_colors[int(l_label * 2 + j)])
                    legend = [Line2D(X, df.iloc[:, body_i], color='k', label='x'),
                              Patch(facecolor=uniform_colors[int(0 * 2 + j)], alpha=0.8, label="sit"),
                              Patch(facecolor=uniform_colors[int(2 * 2 + j)], alpha=0.8, label="stand"),
                              Patch(facecolor=uniform_colors[int(4 * 2 + j)], alpha=0.8, label="walk"),
                              Patch(facecolor=uniform_colors[int(6 * 2 + j)], alpha=0.8, label="lie"),
                              Patch(facecolor=uniform_colors[int(8 * 2 + j)], alpha=0.8, label="interact"),
                              Patch(facecolor=uniform_colors[int(9 * 2 + j)], alpha=0.8, label="wear"),
                              ]
                # plot time sequence and motif segmentation
                x = df.iloc[:, body_i]
                y = df.iloc[:, body_i + 1]
                c = np.linspace(0, 1, len(x))
                t = np.arange(0, len(x))
                axs[fig_i].scatter(t, x, c=c, cmap=plt.get_cmap('plasma'), s=20, label='x')
                axs[fig_i].scatter(t, y, c=c, cmap=plt.get_cmap('viridis'), s=20, label='y')
                # lx = axs[fig_i].plot(x, label='x', c=c, color=plt.get_cmap('plasma'))
                # ly = axs[fig_i].plot(y, label='y', c=c, color=plt.get_cmap('viridis'))


                axs[fig_i].set_title(titles[fig_i])
                axs[fig_i].legend(handles=legend, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)
                axs[fig_i].grid(False)
                axs[fig_i].set_xlabel('time')
                axs[fig_i].set_ylabel('marker position in pixels')
                axs[fig_i].set_ylim([0, 720])


#%% plot trajectory


for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print(v)

        data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))

        dlc_cluster_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\filtered_DLC_{}_km_label_{}.npy'.format(onedrive_path,
                                                                                                        project_name, v,
                                                                                                        n_cluster,
                                                                                                        n_cluster,
                                                                                                        v))

        score_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\score_labels_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, v))
        score_label[score_label < 0] = -1
        # bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
        score_label_merged = score_label.copy()
        score_label_merged[score_label_merged == 1] = 0
        score_label_merged[score_label_merged == 3] = 2
        score_label_merged[score_label_merged == 5] = 4
        score_label_merged[score_label_merged == 7] = 6
        # latent vector has time window of temp_win, thus is temp_win shorter than data
        # we need to crop data to match latent vector
        labels = [vame_label,
                  dlc_cluster_label[temp_win // 2:-temp_win // 2],
                  score_label,
                  score_label_merged,
                  ]

        # a colored line plot for every body keypoint
        for body_i in range(39, 41, 2):
            d = np.concatenate((labels[0].reshape(-1, 1), data[temp_win // 2:-temp_win // 2]), axis=1)
            df = pd.DataFrame(d, columns=dlc_labels)
            x = df.iloc[:, body_i]
            y = df.iloc[:, body_i + 1]
            c = np.linspace(0, 1, len(x))
            t = np.arange(0, len(x))
            # fig, axs = plt.subplots(4, 1, figsize=(12, 12))
            # # plot motif segmentation in three approaches [vame, dlc, score]
            #
            #
            #

            #
            # plt.suptitle('{}-{}-{}'.format(group[j], v, df.columns[body_i][:-1]))
            # plt.show()
            #
            # pwd = r'{}\Behavior_VAE_data\{}\figure\dwell_time_n_dlc'.format(onedrive_path, project_name)
            # Path(pwd).mkdir(parents=True, exist_ok=True)
            # fname = "{}-{}_{}_{}_dwell_time_dlc.png".format(group[j], v, n_cluster, df.columns[body_i][:-1])
            # fname_pdf = "{}-{}_{}_{}_dwell_time_dlc.pdf".format(group[j], v, n_cluster, df.columns[body_i][:-1])
            #
            # fig.savefig(os.path.join(pwd, fname), transparent=True)
            # fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
            #
            # #plot keypoint transparent trajectory over time
            # fig, ax = plt.subplots(figsize=(10, 5))
            # cmap = plt.get_cmap('plasma')
            # c = np.linspace(0, 1, len(x))
            # t = np.arange(0, len(x))
            # scatter_plt = ax.scatter(t, x, c=c, cmap=cmap, s=30)
            # ax.set_xlabel('time')
            # ax.set_ylabel('marker position in pixels')
            # ax.set_ylim([0, 720])
            # ax.grid(False)
            # ax.set_title('{}-{}-{} x position'.format(group[j], v, df.columns[body_i][:-1]))
            # fig.colorbar(scatter_plt)
            # fig.show()
            # pwd = r'{}\Behavior_VAE_data\{}\figure\dwell_time_n_dlc'.format(onedrive_path, project_name)
            # fname = "{}-{}_{}_{}_dwell_time_dlc_plain_xpos.png".format(group[j], v, n_cluster,
            #                                                            df.columns[body_i][:-1])
            # fname_pdf = "{}-{}_{}_{}_trajectory_dlc_plain_xpos.pdf".format(group[j], v, n_cluster,
            #                                                                df.columns[body_i][:-1])
            # fig.savefig(os.path.join(pwd, fname), transparent=True)
            # fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

            # plot spatial trajectory
            fig, ax = plt.subplots(figsize=(10, 5))
            img = matplotlib.image.imread(r'{}\Behavior_VAE_data\DLC raw\background.png'.format(onedrive_path))
            img = cv2.flip(img, 0)
            room = ax.imshow(img)
            cmap = plt.get_cmap('plasma')
            c = np.linspace(0, 1, len(x))
            scatter_plt = ax.scatter(x, y, c=c, cmap=cmap, s=30, alpha=0.4)
            ax.set_ylim([0, 540])
            ax.set_xlim([0, 720])
            ax.grid(False)
            ax.axis('off')
            ax.set_title('{}-{}-{}'.format(group[j], v, df.columns[body_i][:-1]))
            fig.colorbar(scatter_plt)
            fig.show()


            pwd = r'{}\Behavior_VAE_data\{}\figure\dwell_time_n_dlc'.format(onedrive_path, project_name)
            fname = "{}-{}_{}_{}_trajectory_dlc.png".format(group[j], v, n_cluster, df.columns[body_i][:-1])
            fname_pdf = "{}-{}_{}_{}_trajectory_dlc.pdf".format(group[j], v, n_cluster, df.columns[body_i][:-1])
            fig.savefig(os.path.join(pwd, fname), transparent=True)
            fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

#%% Plot latent trajactory overlapping motif segmentations [three approaches]
# Fig. 1d, Fig. 2b, c is here
from sklearn.decomposition import PCA
def map_to_range(lst):
    unique_values = sorted(set(lst))  # Get unique values from the list and sort them
    mapping = {value: i % 10 for i, value in enumerate(unique_values)}  # Map each unique value to 0-9
    mapped_list = [mapping[value] for value in lst]  # Map values in the list to 0-9 based on the mapping
    return mapped_list

titles = ["vame", 'dlc k-means', 'scores', 'scored_merged']
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print(v)

        data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))

        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy'))  # L x 30
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))

        dlc_cluster_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\filtered_DLC_{}_km_label_{}.npy'.format(onedrive_path,
                                                                                                        project_name, v,
                                                                                                        n_cluster,
                                                                                                        n_cluster,
                                                                                                        v))
        dlc_cluster_label = dlc_cluster_label[temp_win // 2:-temp_win // 2]
        mmaction_label = mLabels[v]
        s3d_label = sLabels[v]
        score_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\score_labels_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, v))
        # score_labels has already been aligned with vame labels with time window difference


        score_label[score_label < 0] = -1

        score_label_merged = score_label.copy()
        score_label_merged[score_label_merged == 1] = 0
        score_label_merged[score_label_merged == 3] = 2
        score_label_merged[score_label_merged == 5] = 4
        score_label_merged[score_label_merged == 7] = 6


        # for easier visualization, we only plot the first 15 min
        data = data[temp_win // 2:-temp_win // 2]
        data = data[:15 * 60 * 30]
        vame_label = vame_label[:27000]
        dlc_cluster_label = dlc_cluster_label[:27000]
        score_label = score_label[:27000]
        score_label_merged = score_label_merged[:27000]


        # latent vector has time window of temp_win, thus is temp_win shorter than data
        # we need to crop data to match latent vector
        labels = [vame_label,
                  dlc_cluster_label,
                  score_label,
                  score_label_merged,
                  mmaction_label,
                  s3d_label
                  ]
        # now all these labels are >= 27000

        # a colored line plot for every body keypoint

        fig, axs = plt.subplots(len(labels)+2, 1, figsize=(12, 24)) #4
        # plot motif segmentation in five approaches [vame, dlc, score, mmaction, s3d]
        for fig_i in range(len(labels)+2):
            d = np.concatenate((labels[fig_i].reshape(-1, 1), data), axis=1)

            df = pd.DataFrame(d, columns=dlc_labels)
            cmap_latent = plt.get_cmap('tab20')
            cmap_light = plt.get_cmap('Wistia')
            orange_colors = cmap_light(np.linspace(0, 1, 20))


            cmap = plt.get_cmap('tab20').colors #my_colormap('a') #

            #uniform_colors = cividis_cmap(np.linspace(0, 1, 20))
            uniform_colors = my_colormap('blackbrown10')
            gray_colors = ['#83858c','#3c4e57','#ced7dd','#f4edea', '#000000','#b5838d']#my_colormap('greyscale_categorical6')
            condition = df['label']
            current_c = condition[0]
            # color_dic = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 5}
            labels_key = sorted(set(condition))
            labels_value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            color_dic = dict(zip(labels_key, labels_value))

            spans = [[0, 0]]
            span_label = [current_c]


            for ii, c in enumerate(condition):
                if c != current_c:
                    # change to i-1 if gap between two conditions is to be left empty
                    spans[-1][-1] = ii
                    spans.append([ii, None])
                    span_label.append(c)
                    current_c = c
            # assumes that the last condition is not on its own but same as previous
            x = df.iloc[:, 0]
            X = np.arange(len(x))
            if fig_i == 0 or fig_i == 1 or fig_i == 4 or fig_i == 5:
                spans[-1][-1] = len(condition) - 1
                # span_label.append(condition[-1])
                for iii in range(len(spans)):
                    span = spans[iii]
                    l_label = span_label[iii]
                    if fig_i == 4 or fig_i == 5:
                        l_label = color_dic[int(l_label)]
                    axs[fig_i].axvspan(span[0], span[1], alpha=1, color=cmap[int(l_label * 2 + 0)])

                legend = [Patch(facecolor=cmap[int(0* 2 )], alpha=1, label=0),
                          Patch(facecolor=cmap[int(1* 2 )], alpha=1, label=1),
                          Patch(facecolor=cmap[int(2* 2)], alpha=1, label=2),
                          Patch(facecolor=cmap[int(3* 2 )], alpha=1, label=3),
                          Patch(facecolor=cmap[int(4* 2 )], alpha=1, label=4),
                          Patch(facecolor=cmap[int(5* 2 )], alpha=1, label=5),
                          Patch(facecolor=cmap[int(6* 2)], alpha=1, label=6),
                          Patch(facecolor=cmap[int(7* 2 )], alpha=1, label=7),
                          Patch(facecolor=cmap[int(8* 2 )], alpha=1, label=8),
                          Patch(facecolor=cmap[int(9* 2 )], alpha=1, label=9),
                          ]
            elif fig_i == 2:
                spans[-1][-1] = len(condition) - 1
                # span_label.append(condition[-1])
                for iii in range(len(spans)):
                    span = spans[iii]
                    l_label = span_label[iii]
                    axs[fig_i].axvspan(span[0], span[1], alpha=1, color=uniform_colors[int(l_label)])

                legend = [
                          Patch(facecolor=uniform_colors[int(0)], alpha=0.8, label="sit"),
                          Patch(facecolor=uniform_colors[int(1)], alpha=0.8, label="sit_obj"),
                          Patch(facecolor=uniform_colors[int(2)], alpha=0.8, label="stand"),
                          Patch(facecolor=uniform_colors[int(3)], alpha=0.8, label="stand-obj"),
                          Patch(facecolor=uniform_colors[int(4)], alpha=0.8, label="walk"),
                          Patch(facecolor=uniform_colors[int(5)], alpha=0.8, label="walk_obj"),
                          Patch(facecolor=uniform_colors[int(6)], alpha=0.8, label="lie"),
                          Patch(facecolor=uniform_colors[int(7)], alpha=0.8, label="lie_obj"),
                          Patch(facecolor=uniform_colors[int(8)], alpha=0.8, label="interact"),
                          Patch(facecolor=uniform_colors[int(9)], alpha=0.8, label="wear"),
                          ]
            elif fig_i == 3:
                spans[-1][-1] = len(condition) - 1
                # span_label.append(condition[-1])
                for iii in range(len(spans)):
                    span = spans[iii]
                    l_label = span_label[iii]
                    color_idx = color_dic[int(l_label)]
                    axs[fig_i].axvspan(span[0], span[1], alpha=1, color=gray_colors[color_idx])
                legend = [
                          Patch(facecolor=gray_colors[int(0)], alpha=0.8, label="sit"),
                          Patch(facecolor=gray_colors[int(1)], alpha=0.8, label="stand"),
                          Patch(facecolor=gray_colors[int(2)], alpha=0.8, label="walk"),
                          Patch(facecolor=gray_colors[int(3)], alpha=0.8, label="lie"),
                          Patch(facecolor=gray_colors[int(4)], alpha=0.8, label="interact"),
                          Patch(facecolor=gray_colors[int(5)], alpha=0.8, label="wear"),
                          ]
            axs[fig_i].set_title(titles[fig_i])
            axs[fig_i].legend(handles=legend, bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1)
            axs[fig_i].grid(False)
            # axs[fig_i].set_xlabel('time')
            axs[fig_i].set_ylabel('latent (d = 10)')
            axs[fig_i].set_ylim([0, 720])


        # plot latents and motif segmentation
        copper_cmap = plt.get_cmap('copper')
        copper_colors = copper_cmap(np.linspace(0, 1, 20))
        df1 = pd.DataFrame(latent_vector[:15 * 60 * 30])

        import numpy as np
        from sklearn.linear_model import LinearRegression
        X = latent_vector[:15 * 60 * 30]
        y = data[:, body_i].reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        score = reg.score(X, y)

        for latent_i in range(10):
            x = df1.iloc[:, latent_i]+200*latent_i
            c = np.linspace(0, 1, len(x))
            lx = axs[-3].plot(x, color=copper_colors[int(latent_i * 2 + j)], label=latent_i)
        axs[-3].set_title(f"Regression Latent to center feet x score {score}")
        axs[-3].grid(False)

        # regression of the latent to the keypoints



        # plot keypoints
        body_i = 39  # center_feetx
        x = df.iloc[:, body_i]
        y = df.iloc[:, body_i + 1]
        c = np.linspace(0, 1, len(x))
        t = np.arange(0, len(x))
        axs[-2].scatter(t, x, c=c, cmap=plt.get_cmap('viridis'), s=5, label='x')
        axs[-2].scatter(t, y, c=c, cmap=plt.get_cmap('plasma'), s=5, label='y')
        axs[-2].grid(False)


        #plot keypoints over ethogram
        body_i = 39  # center_feetx
        x = df.iloc[:, body_i]
        y = df.iloc[:, body_i + 1]
        c = np.linspace(0, 1, len(x))
        t = np.arange(0, len(x))

        spans[-1][-1] = len(condition) - 1
        # span_label.append(condition[-1])
        for iii in range(len(spans)):
            span = spans[iii]
            l_label = span_label[iii]
            color_idx = color_dic[int(l_label)]
            axs[-1].axvspan(span[0], span[1], alpha=1, color=gray_colors[color_idx])
        axs[-1].scatter(t, x, c=c, cmap=plt.get_cmap('spring'), s=5, label='x')
        axs[-1].grid(False)



        plt.suptitle('{}-{}'.format(group[j], v))
        plt.show()

        pwd = r'{}\Behavior_VAE_data\{}\figure\dwell_time_n_latent'.format(onedrive_path, project_name)
        Path(pwd).mkdir(parents=True, exist_ok=True)
        fname = "{}-{}_{}_dwell_time_latent.png".format(group[j], v, n_cluster)
        fname_pdf = "{}-{}_{}_dwell_time_latent.pdf".format(group[j], v, n_cluster)
        fig.savefig(os.path.join(pwd, fname), transparent=True)
        fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

        plt.close('all')


#%%  OLD STUFF
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
from plotting.get_paths import get_my_path
#%%
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']
data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
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
