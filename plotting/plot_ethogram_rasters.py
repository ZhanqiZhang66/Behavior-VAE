# Created by zhanq at 5/23/2024
# File:
# Description:
# Scenario:
# Usage
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
#%% Plot latent trajactory overlapping motif segmentations [three approaches]
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
def map_to_range(lst):
    unique_values = sorted(set(lst))  # Get unique values from the list and sort them
    mapping = {value: i % 10 for i, value in enumerate(unique_values)}  # Map each unique value to 0-9
    mapped_list = [mapping[value] for value in lst]  # Map values in the list to 0-9 based on the mapping
    return mapped_list
def intensity_estimation(labels):

    # Initialize the binary list with 0s
    events = [0] * len(labels)
    # Iterate through the list of labels and mark transitions
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            events[i] = 1
    # Convert to numpy array for convenience
    event_array = np.array(events)

    # Define time steps
    time_steps = np.arange(len(events)).reshape(-1, 1)

    # Find the indices where events occur
    event_indices = np.where(event_array == 1)[0].reshape(-1, 1)

    # Perform kernel density estimation on the event indices
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(event_indices)
    log_density = kde.score_samples(time_steps)
    intensity_estimates = np.exp(log_density)

    # Normalize the intensity estimates if necessary
    intensity_estimates /= np.max(intensity_estimates)
    intensity_estimates *= np.mean(event_array) / np.mean(intensity_estimates)
    return event_array, time_steps, intensity_estimates
#%% stacked ethograms
titles = ["vame"]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0

    rasters = []

    fig, axs = plt.subplots(25, 1, figsize=(15, 5 * 25))
    for i in range(len(videos)):
        ax1 = axs[i]
        v = videos[i]
        print(v)

        data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))

        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy'))  # L x 30
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))


        # for easier visualization, we only plot the first 15 min
        data = data[temp_win // 2:-temp_win // 2]
        data = data[:15 * 60 * 30]
        vame_label = vame_label[:27000]
        event_array, time_steps, intensity_estimates = intensity_estimation(vame_label)
        rasters.append(event_array)
        # latent vector has time window of temp_win, thus is temp_win shorter than data
        # we need to crop data to match latent vector
        labels = vame_label
        d = np.concatenate((labels.reshape(-1, 1), data), axis=1)

        df = pd.DataFrame(d, columns=dlc_labels)

        cmap = plt.get_cmap('tab20').colors  # my_colormap('a') #
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

        spans[-1][-1] = len(condition) - 1
        # span_label.append(condition[-1])
        for iii in range(len(spans)):
            span = spans[iii]
            l_label = span_label[iii]

            ax1.axvspan(span[0], span[1], alpha=1, color=cmap[int(l_label * 2 + 0)])

        legend = [Patch(facecolor=cmap[int(0 * 2)], alpha=1, label=0),
                  Patch(facecolor=cmap[int(1 * 2)], alpha=1, label=1),
                  Patch(facecolor=cmap[int(2 * 2)], alpha=1, label=2),
                  Patch(facecolor=cmap[int(3 * 2)], alpha=1, label=3),
                  Patch(facecolor=cmap[int(4 * 2)], alpha=1, label=4),
                  Patch(facecolor=cmap[int(5 * 2)], alpha=1, label=5),
                  Patch(facecolor=cmap[int(6 * 2)], alpha=1, label=6),
                  Patch(facecolor=cmap[int(7 * 2)], alpha=1, label=7),
                  Patch(facecolor=cmap[int(8 * 2)], alpha=1, label=8),
                  Patch(facecolor=cmap[int(9 * 2)], alpha=1, label=9),
                  ]


        ax1.grid(False)
        # axs[fig_i].set_xlabel('time')
        ax1.set_ylim([0, 1])

        # ax2 = ax1.twinx()
        # color = 'tab:red'
        # ax2.set_ylabel('Poisson Events', color=color)
        # ax2.plot(time_steps, intensity_estimates, label='Estimated Intensity Function', color='k')
        # ax2.scatter(time_steps, event_array, label='Event Data', color='red', zorder=3)
        # ax2.tick_params(axis='y', labelcolor=color)

    # Adjust layout
    plt.tight_layout()
    plt.show()


#%% raster
titles = ["vame"]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0

    rasters = []

    fig, axs = plt.subplots(25, 1, figsize=(15, 5 * 25))
    for i in range(len(videos)):
        ax1 = axs[i]
        v = videos[i]
        print(v)

        data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))

        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy'))  # L x 30
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))


        # for easier visualization, we only plot the first 15 min
        data = data[temp_win // 2:-temp_win // 2]
        data = data[:15 * 60 * 30]
        vame_label = vame_label[:27000]
        event_array, time_steps, intensity_estimates = intensity_estimation(vame_label)
        rasters.append(event_array)


    # rasters
    plt.figure(figsize=(10, 2))
    rasters_array = np.array(rasters)
    plt.matshow(rasters_array, aspect='auto', cmap='Greys', interpolation='none')
    plt.title(f'{group[j]}')
    plt.show()







