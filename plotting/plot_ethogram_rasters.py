# Created by zhanq at 5/23/2024
# File:
# Description:
# Scenario:
# Usage
#%%
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pylab as pl
import os

import tqdm
from pathlib import Path

from sklearn.cluster import KMeans

import pandas as pd
import glob
import datetime

from pathlib import Path
import pathlib
from data.load_data import load_pt_data
from analysis.Classifiers.Generation.utils import my_colormap
from plotting.get_paths import get_my_path
from itertools import combinations
import seaborn as sns
import pandas as pd
import pymannkendall as mk
import random
from scipy import stats
#%%
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']

#%% Define Project
project_name = 'BD25-HC25-final-May17-2023'
project_path = f'{onedrive_path}\Behavior_VAE_data\{project_name}'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
dlc_path = os.path.join(project_path,"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
model_name = 'VAME'


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

csv_path = os.path.join(project_path,"videos","pose_estimation")
confidence = 0.9
group = ['HC','BD']
temp_win = 60
#%% Plot latent trajactory overlapping motif segmentations [three approaches]
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
def map_to_range(lst):
    unique_values = sorted(set(lst))  # Get unique values from the list and sort them
    mapping = {value: i % 10 for i, value in enumerate(unique_values)}  # Map each unique value to 0-9
    mapped_list = [mapping[value] for value in lst]  # Map values in the list to 0-9 based on the mapping
    return mapped_list
def label_to_transition(label):
    events = [0] * len(label)
    for i in range(1, len(label)):
        if label[i] != label[i - 1]:
            events[i] = 1
    return np.array(events)
def intensity_estimation(labels, bandwidth):

    events = label_to_transition(labels)
    event_array = np.array(events)

    # Define time steps
    time_steps = np.arange(len(events)).reshape(-1, 1)

    # Find the indices where events occur
    event_indices = np.where(event_array == 1)[0].reshape(-1, 1)

    # Perform kernel density estimation on the event indices
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(event_indices)
    log_density = kde.score_samples(time_steps)
    intensity_estimates = np.exp(log_density)

    # Normalize the intensity estimates if necessary
    intensity_estimates /= np.max(intensity_estimates)
    intensity_estimates *= np.mean(event_array) / np.mean(intensity_estimates)
    return event_array, time_steps, intensity_estimates
def plot_ethogram(labels, data, ax, alpha=1, show_event=False):

    events = label_to_transition(labels)
    # # Mark the change points
    if show_event:
        for k, cp in enumerate(events):
            if cp == 1:
                ax.axvline(x=k, color='k', linestyle='-')

    # latent vector has time window of temp_win, thus is temp_win shorter than data
    # we need to crop data to match latent vector
    d = np.concatenate((labels.reshape(-1, 1), data), axis=1)

    df = pd.DataFrame(d, columns=dlc_labels)

    cmap = plt.get_cmap('tab20').colors

    # this is motif label
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

    spans[-1][-1] = len(condition) - 1
    # span_label.append(condition[-1])
    for iii in range(len(spans)):
        span = spans[iii]
        l_label = span_label[iii]

        ax1.axvspan(span[0], span[1], color=cmap[int(l_label * 2 + 0)], alpha=alpha)

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

    ax.grid(False)
    # axs[fig_i].set_xlabel('time')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(labels)])
def binary_list_to_index_list(binary_list):
  index_list = []
  for i, value in enumerate(binary_list):
    if value:
      index_list.append(i)
  return index_list
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

        data = np.load(os.path.join(project_path, 'data', 'pose_sequence', v + '-90pct_seq.npy'))

        folder = os.path.join(project_path, "results", v, model_name, 'kmeans-' + str(n_cluster), "")
# L x 30
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))


        # for easier visualization, we only plot the first 15 min
        data = data[temp_win // 2:-temp_win // 2]
        data = data[:15 * 60 * 30]
        vame_label = vame_label[:27000]
        plot_ethogram(vame_label, data, ax1, alpha=1, show_event=True)



        # ax2 = ax1.twinx()
        # color = 'tab:red'
        # ax2.set_ylabel('Poisson Events', color=color)
        # ax2.plot(time_steps, intensity_estimates, label='Estimated Intensity Function', color='k')
        # ax2.scatter(time_steps, event_array, label='Event Data', color='red', zorder=3)
        # ax2.tick_params(axis='y', labelcolor=color)

    # Adjust layout
    plt.tight_layout()
    plt.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\raster_ethogram'.format(onedrive_path, project_name)
    Path.mkdir(Path(pwd), exist_ok=True, parents=True)
    fname = "{}-ethogram.png".format(group[j])
    fname_pdf = "{}-ethogram.pdf".format(group[j])
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

#%% raster
bandwidth = 90
titles = ["vame"]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    rasters = []
    for i in range(len(videos)):
        v = videos[i]
        print(v)
        data = np.load(os.path.join(project_path, 'data', 'pose_sequence', v + '-90pct_seq.npy'))

        folder = os.path.join(project_path, "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy'))  # L x 30
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))

        data = data[temp_win // 2:-temp_win // 2]
        data = data[:15 * 60 * 30]
        vame_label = vame_label[:27000]
        event_array, time_steps, intensity_estimates = intensity_estimation(vame_label, bandwidth=bandwidth)
        rasters.append(binary_list_to_index_list(event_array))
    raster_to_plot = np.array(rasters)
    fig, ax1 = plt.subplots(1, 1, figsize=(30, 5))
    ax1.eventplot(raster_to_plot, orientation='horizontal', colors='black', lineoffsets=1, linelengths=1, linestyles='solid')
    ax1.invert_yaxis()
    plt.title(f'{group[j]}')
    plt.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\raster_ethogram'.format(onedrive_path, project_name)
    Path.mkdir(Path(pwd), exist_ok=True, parents=True)
    fname = "{}-rasters.png".format(group[j])
    fname_pdf = "{}-rasters.pdf".format(group[j])
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%%
#%% estimated intensity function
titles = ["vame"]
intensity_estimates_list = [[],[]]
event_array_list = [[],[]]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    intensity_estimates_j = []
    event_array_j = []
    for i in range(len(videos)):

        v = videos[i]
        print(v)

        data = np.load(os.path.join(project_path, 'data', 'pose_sequence', v + '-90pct_seq.npy'))

        folder = os.path.join(project_path, "results", v, model_name, 'kmeans-' + str(n_cluster), "")
# L x 30
        labels = []
        vame_label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, n_cluster, v))


        # for easier visualization, we only plot the first 15 min
        data = data[temp_win // 2:-temp_win // 2]
        data = data[:15 * 60 * 30]
        vame_label = vame_label[:27000]

        event_array, time_steps, intensity_estimates = intensity_estimation(vame_label, bandwidth=900)

        intensity_estimates_j.append(intensity_estimates)
        event_array_j.append(event_array)
        # fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        # ax1 = axs
        # plot_ethogram(vame_label, data, ax1, alpha=1, show_event=False)
        #
        #
        #
        # ax2 = ax1.twinx()
        # color = 'tab:red'
        # ax2.set_ylabel('Poisson Events', color=color)
        # ax2.plot(time_steps, intensity_estimates, label='Estimated Intensity Function', color='blue')
        # for i,event in enumerate(event_array):
        #     if event == 1:
        #         ax2.scatter(i, event*0.1, label='Event Data', color='k', zorder=3)
        # ax2.tick_params(axis='y', labelcolor=color)
        #
        # plt.tight_layout()
        # plt.show()
    intensity_estimates_list[j] = intensity_estimates_j
    event_array_list[j] = event_array_j
#%% plot the mean of intensity_estimation

fig, ax = plt.subplots(1, 1, figsize=(15, 5))
b_o_colors = ['#1f77b4', '#ff7f0e']
for j in range(2):
    intensity_estimates_ = np.array(intensity_estimates_list[j])
    avg_intensity = intensity_estimates_.mean(axis=0)
    std_intensity = intensity_estimates_.std(axis=0)
    y = avg_intensity
    error_y = std_intensity

    x = np.arange(27000)
    ax.fill_between(x, y - error_y, y + error_y, norm=plt.Normalize(vmin=0, vmax=9),
                    alpha=0.2, facecolor=b_o_colors[j])
    ax.plot(x, y, color=b_o_colors[j], zorder=1)
plt.xlim([0, 27000])
plt.title(f'BD vs control(blue)')
plt.show()
#%%  plot mean of sum of event_array_list
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
b_o_colors = ['#1f77b4', '#ff7f0e']
for j in range(2):
    event_array_list_ = np.array(event_array_list[j])
    avg_event = event_array_list_.mean(axis=0)
    std_intensity = event_array_list_.std(axis=0)
    y = avg_event
    error_y = std_intensity

    x = np.arange(27000)
    ax.fill_between(x, y - error_y, y + error_y, norm=plt.Normalize(vmin=0, vmax=9),
                    alpha=0.2, facecolor=b_o_colors[j])
    ax.plot(x, y, color=b_o_colors[j], zorder=1)
plt.xlim([0, 27000])
plt.title(f'BD vs control(blue)')
plt.show()
#%% bootstrap the intensity_estimates
from tsbootstrap import MovingBlockBootstrap
rng = random.seed(9)
for j in range(2):
    X_bootstrapped = []
    for sample in range(25):
        intensity_estimates_ = np.array(intensity_estimates_list[j][sample]).transpose()
        # Instantiate the bootstrap object

        block_length = 27000
        rng = 42
        mbb = MovingBlockBootstrap(n_bootstraps=100, rng=rng, block_length=block_length)

        # Generate bootstrapped samples
        bootstrapped_samples = mbb.bootstrap(intensity_estimates_, return_indices=False)

        # Collect bootstrap samples

        for data in bootstrapped_samples:
            X_bootstrapped.append(data)

    X_bootstrapped = np.array(X_bootstrapped)

    # plot the bootstrapped intensity_estimates
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    for i in range(250):
        y = X_bootstrapped[i]
        ax.plot(np.arange(27000), y, color='blue', alpha=0.1)
    for ii in range(25):
        ax.plot(np.arange(27000), np.array(intensity_estimates_list[j][ii]), color='red', alpha=0.1)

    ax.plot(np.arange(27000), np.mean(X_bootstrapped, axis=0), color='blue', zorder=1)
    ax.plot(np.arange(27000), np.array(intensity_estimates_list[j]).mean(axis=0), color='red', zorder=1)
    plt.show()


#%%

def compute_row_differences(A, B):

    num_rows = A.shape[0]

    # Initialize a list to store the differences
    differences = []

    # Compute differences between each row in A and each row in B
    for i, j in combinations(range(num_rows), 2):
        # Compute the difference for the pair (i, j)
        diff_ij = A[i] - B[j]
        differences.append(diff_ij)

        # Compute the difference for the pair (j, i)
        diff_ji = A[j] - B[i]
        differences.append(diff_ji)

    # Convert the list of differences to a numpy array for easier manipulation
    differences = np.array(differences)

    return differences
def compute_auc(y):
    return np.trapz(y, x=np.arange(len(y)))
# plot the difference between the two groups


auc_epoch = []
auc_shuffle_epoch = []
diff_epoch = []
diff_shuffle_epoch = []

for epoch in range(1, 4):
    # split the data evenly in three epochs
    HC_intensity_estimates_ = np.array(intensity_estimates_list[0])
    BD_intensity_estimates_ = np.array(intensity_estimates_list[1])
    HC_intensity_estimates_ = HC_intensity_estimates_[:, (epoch-1)*9000:epoch*9000]
    BD_intensity_estimates_ = BD_intensity_estimates_[:, (epoch-1)*9000:epoch*9000]

    difference = compute_row_differences(HC_intensity_estimates_, BD_intensity_estimates_)
    diff_epoch.append(difference)
    # shuffled
    intensity_estimates_list_all = np.concatenate((HC_intensity_estimates_, BD_intensity_estimates_), axis=0).tolist()
    random.shuffle(intensity_estimates_list_all)
    HC_intensity_estimates_random = np.array(intensity_estimates_list_all[:25])
    BD_intensity_estimates_random = np.array(intensity_estimates_list_all[25:])
    difference_shuffle = compute_row_differences(HC_intensity_estimates_random, BD_intensity_estimates_random)
    diff_shuffle_epoch.append(difference_shuffle)
    aucs = []
    aucs_shuffle = []
    for i in range(difference.shape[0]):
        y = difference[i]
        y_shuffled = difference_shuffle[i]
        x = np.arange(9000)
        auc = np.trapz(y, x)
        auc_shuffle = np.trapz(y_shuffled, x)
        aucs.append(auc)
        aucs_shuffle.append(auc_shuffle)
    auc_epoch.append(aucs)
    auc_shuffle_epoch.append(aucs_shuffle)

# do t-test on diff_epoch and diff_shuffle_epoch
difference_overall = compute_row_differences(np.array(intensity_estimates_list[0]), np.array(intensity_estimates_list[1]))
stats.ttest_1samp(np.array(difference_overall).mean(axis=1), popmean=0)
stats.ttest_ind(np.array(intensity_estimates_list[0]).flatten(), np.array(intensity_estimates_list[1]).flatten())

# make a histogram for intensity_estimates_list[0] and intensity_estimates_list[1]
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
sns.histplot(np.array(intensity_estimates_list[0]).flatten(), color='blue', ax=ax)
sns.histplot(np.array(intensity_estimates_list[1]).flatten(), color='red', ax=ax)
# do not show 0
plt.xlim([0.0011, 0.2])
plt.show()



auc_epoch_ = np.array(auc_epoch).flatten()
auc_shuffle_epoch_ = np.array(auc_shuffle_epoch).flatten()
# for each epoch, plot the auc of the difference between two groups
df = pd.DataFrame(auc_epoch_.T, columns=['Data'])
df['Epoch'] = np.concatenate([np.ones(600)*1, np.ones(600)*2, np.ones(600)*3])
df['Type'] = 'Data'

df_shuffle = pd.DataFrame(auc_shuffle_epoch_.T, columns=['Data'])
df_shuffle['Epoch'] = np.concatenate([np.ones(600)*1, np.ones(600)*2, np.ones(600)*3])
df_shuffle['Type'] = 'Shuffled'
# plot shuffled violin next to data violin
df_all = pd.concat([df, df_shuffle])
sns.violinplot(x='Epoch', y='Data', hue='Type', data=df_all)
plt.show()

x_data = np.mean(np.array(auc_epoch), axis=1)
x_shuffle = np.mean(np.array(auc_shuffle_epoch), axis=1)
# fit linear regression to x_data and x_shuffle
x = np.arange(3)
slope, b1, r, p, se = stats.linregress(x, x_data)
slope2, b2, r2, p2, se2 = stats.linregress(x, x_shuffle)
t = (slope - slope2) / np.sqrt(abs(se - se2))
p = stats.t.sf(abs(t), df=len(x)+len(x)-4)
print(p)

#%% Statistical test

#
# HC_intensity_estimates_ = np.array(intensity_estimates_list[0])
# BD_intensity_estimates_ = np.array(intensity_estimates_list[1])
# HC_avg_intensity = HC_intensity_estimates_.mean(axis=0)
# BD_avg_intensity = BD_intensity_estimates_.mean(axis=0)
# print(mk.original_test(HC_avg_intensity))
# print(mk.original_test(BD_avg_intensity))
# x = np.arange(27000)
# slope, b1, r, p, se = stats.linregress(x, HC_avg_intensity)
# slope2, b2, r2, p2, se2 = stats.linregress(x, BD_avg_intensity)
# t = (slope - slope2) / np.sqrt(se - se2)
# p = stats.t.sf(abs(t), df=len(x)+len(x)-4)
# print(p)
#
# intensity_estimates_list_all = np.concatenate((HC_intensity_estimates_, BD_intensity_estimates_), axis=0).tolist()
# random.shuffle(intensity_estimates_list_all)
# HC_intensity_estimates_random = np.array(intensity_estimates_list_all[:25])
# BD_intensity_estimates_random = np.array(intensity_estimates_list_all[25:])
# HC_avg_intensity_random = HC_intensity_estimates_random.mean(axis=0)
# BD_avg_intensity_random = BD_intensity_estimates_random.mean(axis=0)
# print(mk.original_test(HC_avg_intensity_random))
# print(mk.original_test(BD_avg_intensity_random))
# slope3, b3, r3, p3, se3 = stats.linregress(x, HC_avg_intensity_random)
# slope4, b4, r2, p4, se4 = stats.linregress(x, BD_avg_intensity_random)
# t2 = (slope3 - slope4) / np.sqrt(se3 - se4)
# p2 = stats.t.sf(abs(t2), df=len(x)+len(x)-4)
# print(p2)
#%% moving average on transition
#%% estimated intensity function

# def moving_average(data, window_size):
#     return np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), 'valid') / window_size, axis=0, arr=data)
# titles = ["vame"]
# for j, videos in enumerate([control_videos, BD_videos]):
#     n = 0
#     fig, axs = plt.subplots(1, 1, figsize=(15, 5))
#     events = []
#     for i in range(len(videos)):
#
#         v = videos[i]
#         print(v)
#
#         data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))
#
#         folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
# # L x 30
#         labels = []
#         vame_label = np.load(
#             r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
#                 onedrive_path, project_name, v, n_cluster, n_cluster, v))
#
#
#         # for easier visualization, we only plot the first 15 min
#         data = data[temp_win // 2:-temp_win // 2]
#         data = data[:15 * 60 * 30]
#         vame_label = vame_label[:27000]
#
#         event_array, time_steps, intensity_estimates = intensity_estimation(vame_label, bandwidth=bandwidth)
#         events.append(event_array)
#
#     events = np.array(events)
#     avg_transition = moving_average(events, window_size=bandwidth)
#     plt.plot(range(len(avg_transition)), avg_transition.mean(axis=0))
#     plt.show()





