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
from pathlib import Path
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
from sklearn.decomposition import PCA
from data.load_data import load_pt_data
from plotting.get_paths import get_my_path
#%%
load_precomputed_sliding_window = False
n_cluster = 10
#%%
#%%
if not load_precomputed_sliding_window:
    def count_zeros(transition_m):
        transition = transition_m.copy()
        zero_rows = np.all(transition == 0, axis=1)
        zero_rows_i = np.where(zero_rows == True)
        zero_cols = np.all(transition == 0, axis=0)
        return len(zero_rows_i[0]), np.count_nonzero(transition == 1), np.count_nonzero(transition == 0)
    def add_self_transition(transition_m, last_state):
        transition = transition_m.copy()
        zero_rows = np.all(transition == 0, axis=1)
        zero_rows_i = np.where(zero_rows == True)
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


    def effective_num_states(transtion_m):
        effective_num_every_state = []
        for row in transtion_m:
            sum_p_ij = np.sum(np.square(row))
            if sum_p_ij == 0:
                effective_num_every_state.append(0)

            else:
                effective_num_every_state.append(1 / sum_p_ij)
        effective_num_avg = np.mean(effective_num_every_state)
        return effective_num_every_state, effective_num_avg

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
    #%% Sliding window of 3 min analysis


    #%%
    myPath = get_my_path()
    onedrive_path = myPath['onedrive_path']
    github_path = myPath['github_path']
    data_path = myPath['data_path']
    #%%
    project_name = 'BD25-HC25-final-May17-2023'
    project_path = f'{onedrive_path}\Behavior_VAE_data\{project_name}'
    config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path,
                                                           project_name)  # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
    cfg = read_config(config)
    dlc_path = os.path.join(project_path, "videos", "\pose_estimation")
    cfg = read_config(config)
    dlc_path = os.path.join(cfg['project_path'], "videos",
                            "\pose_estimation")  # dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)

    n_scores = 11
    model_name = 'VAME'
    cluster_start = cfg['time_window'] / 2

    d_latent = 10
    window_size = int(3 * 60 * 30)
    five_min_frame_no = 5 * 60 * 30
    # %%
    b_o_colors = ['#1f77b4', '#ff7f0e']

    data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
    control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
    BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
    score_bahavior_names = ["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact",
                            "wear", "exercise"]
    n_subject_in_population = len(control_videos)

    titles = ["CP", "BD"]

    slide_window = {
        "subject": [],
        "start_frame": [],
        "is_BD": [],
        # "entropy": [],
        # "effective_num_every_state": [],
        # "effective_num_avg": [],
        # "num_zero_row":[],
        # "num_one_item": [],
        # "num_zero_item":[],
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
        # "latent_volume_all_motifs": [],
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
        # "entropy_score": [],
        # "effective_num_every_state_score": [],
        # "effective_num_avg_score": [],
        # "num_zero_row_score": [],
        # "num_one_item_score": [],
        # "num_zero_item_score": [],
        # "motif0_usage_freq_score": [],
        # "motif1_usage_freq_score": [],
        # "motif2_usage_freq_score": [],
        # "motif3_usage_freq_score": [],
        # "motif4_usage_freq_score": [],
        # "motif5_usage_freq_score": [],
        # "motif6_usage_freq_score": [],
        # "motif7_usage_freq_score": [],
        # "motif8_usage_freq_score": [],
        # "motif9_usage_freq_score": [],
        # "entropy_ctl": [],
        # "effective_num_every_state_ctl": [],
        # "effective_num_avg_ctl": [],
        # "num_zero_row_ctl": [],
        # "num_one_item_ctl": [],
        # "num_zero_item_ctl": [],
        # "motif0_usage_freq_ctl": [],
        # "motif1_usage_freq_ctl": [],
        # "motif2_usage_freq_ctl": [],
        # "motif3_usage_freq_ctl": [],
        # "motif4_usage_freq_ctl": [],
        # "motif5_usage_freq_ctl": [],
        # "motif6_usage_freq_ctl": [],
        # "motif7_usage_freq_ctl": [],
        # "motif8_usage_freq_ctl": [],
        # "motif9_usage_freq_ctl": [],
    }
    for i in range(n_cluster):
        slide_window['motif{}_usage_freq'.format(i)] = []
        # slide_window['motif{}_usage_freq_ctl'.format(i)] = []
        # slide_window['motif{}_usage_freq_score'.format(i)] = []
        slide_window['latent_volume_motif{}'.format(i)] = []
    csv_path = os.path.join(cfg['project_path'],"videos","pose_estimation")
#%%
    '''
    Step 1: Mean-centered the population-latent vectors for computing
    '''
    Latent_vectors = [[], []]
    # Get the latent vector of each video
    for j, videos in enumerate([control_videos, BD_videos]):
        n = 0
        for i in range(len(videos)):
            v = videos[i]
            print("Loading {} data...".format(v))

            label = np.load(
                r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path,
                                                                                               project_name, v,
                                                                                               n_cluster, n_cluster, v))

            folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
            latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy'))  # T x d = {10, 30}
            latent_vector = latent_vector[: five_min_frame_no * 3]
            label = label[: five_min_frame_no * 3]
            if i == 0:
                l = label
                latent = latent_vector
            else:
                latent = np.concatenate([latent, latent_vector])
                l = np.concatenate([l, label])

        Latent_vectors[j] = latent


    # compute the mean of latent population
    population_latent_vector = np.vstack(Latent_vectors)
    population_latent_vector_centroid = np.mean(population_latent_vector, axis=0)
    Latent_centroids = population_latent_vector_centroid

    #%%
    pca = PCA(n_components=3)
    for j, videos in enumerate([control_videos, BD_videos]):
        for i in range(len(videos)):
            start = time.time()
            v = videos[i]
            print("Loading {} data...".format(v))
            folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
            label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path,project_name, v,n_cluster,n_cluster,v))
            latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 30
            latent_vector = latent_vector - Latent_centroids

            #
            # control_label = np.load(
            #     r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_km_label_{}.npy'.format(onedrive_path,
            #                                                                                        project_name, v,
            #                                                                                        n_cluster, n_cluster,
            #                                                                                        v))
            # control_transition = compute_transition_matrices([v], [control_label], n_cluster)[0]
            # score_label = np.load(
            #     r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\score_labels_{}.npy'.format(onedrive_path,
            #                                                                                     project_name, v,
            #                                                                                     n_cluster, v))
            # score_label = score_label[: 27000]
            # score_transition = compute_transition_matrices([v], [score_label], n_cluster)[0]

            door_close_time = int(start_frame[v][0])
            start_time = door_close_time

            offset = 0# int(door_close_time - start_time)



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
                # num_zero_row, num_one_item, num_zero_item = count_zeros(window_transition_matrix[0])
                # entropy = compute_l0_entropy(window_transition_matrix[0], window_label[-1])
                # effective_num_every_state, effective_num_avg = effective_num_states(window_transition_matrix[0])

                # num_zero_row_score, num_one_item_score, num_zero_item_score = count_zeros(score_transition)
                # entropy_score = compute_l0_entropy(score_transition, control_label[-1])
                # effective_num_every_state_score, effective_num_avg_score = effective_num_states(score_transition)
                # control_motif_usage = get_motif_usage(control_label, n_cluster)
                #
                # num_zero_row_ctl, num_one_item_ctl, num_zero_item_ctl = count_zeros(control_transition)
                # entropy_ctl = compute_l0_entropy(control_transition, control_label[-1])
                # effective_num_every_state_ctl, effective_num_avg_ctl = effective_num_states(control_transition)
                # score_motif_usage = get_motif_usage(score_label, n_scores)
                #velocity = compute_velocity(data_mat[offset + k: window_size + offset + k], window_size)

                slide_window["subject"].append(v)
                slide_window["start_frame"].append(k)
                slide_window["is_BD"].append(j)
                # slide_window["entropy"].append(entropy)
                # slide_window["num_zero_row"].append(num_zero_row)
                # slide_window["num_one_item"].append(num_one_item)
                # slide_window["num_zero_item"].append(num_zero_item)
                # slide_window["effective_num_every_state"].append(effective_num_every_state)
                # slide_window["effective_num_avg"].append(effective_num_avg)

                # slide_window["entropy_score"].append(entropy_score)
                # slide_window["num_zero_row_score"].append(num_zero_row_score)
                # slide_window["num_one_item_score"].append(num_one_item_score)
                # slide_window["num_zero_item_score"].append(num_zero_item_score)
                # slide_window["effective_num_every_state_score"].append(effective_num_every_state_score)
                # slide_window["effective_num_avg_score"].append(effective_num_avg_score)

                # slide_window["entropy_ctl"].append(entropy_ctl)
                # slide_window["num_zero_row_ctl"].append(num_zero_row_ctl)
                # slide_window["num_one_item_ctl"].append(num_one_item_ctl)
                # slide_window["num_zero_item_ctl"].append(num_zero_item_ctl)
                # slide_window["effective_num_every_state_ctl"].append(effective_num_every_state_score)
                # slide_window["effective_num_avg_ctl"].append(effective_num_avg_ctl)
                for i in range(n_cluster):
                    slide_window['motif{}_usage_freq'.format(i)].append(window_motif_usage[i]/np.sum(window_motif_usage))
                    # slide_window['motif{}_usage_freq_ctl'.format(i)].append(control_motif_usage[i] / np.sum(control_motif_usage))
                    # slide_window['motif{}_usage_freq_score'.format(i)].append(score_motif_usage[i] / np.sum(score_motif_usage))
                # slide_window["motif_usage_freq"].append(window_motif_usage/np.sum(window_motif_usage))

                # K = np.cov(window_latent_vector.T)
                # volume_of_all = np.trace(K)
                # slide_window["latent_volume_all_motifs"].append(volume_of_all)

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

                # slide_window["latent_volume_per_motif"].append(latent_volume_per_motif)
            end = time.time()
            print(f"Runtime of one video is {end - start}")
    pwd = r'{}\Behavior_VAE_data\{}\data\slide_window3_{}motifs_new_motif_volume.csv'.format(onedrive_path,project_name, n_cluster)
    ds_new = pd.DataFrame.from_dict(slide_window)
    ds_new.to_csv(pwd)
#%%
if load_precomputed_sliding_window:
    project_name = 'BD25-HC25-final-May17-2023'
    pwd = r'{}\Behavior_VAE_data\{}\data\slide_window3_{}motifs_new_motif_volume.csv'.format(onedrive_path,project_name, n_cluster)
    ds = pd.read_csv(pwd)


    project_path = f'{onedrive_path}\Behavior_VAE_data\{project_name}'
    config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path,
                                                           project_name)  # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
    cfg = read_config(config)
    dlc_path = os.path.join(project_path, "videos",
                            "\pose_estimation")  # dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)

    model_name = 'VAME'
    cluster_start = cfg['time_window'] / 2
    d_latent = 10
    window_size = int(3 * 60 * 30)
    data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
    control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
    BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
    score_bahavior_names = ["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact",
                            "wear", "exercise"]
    n_subject_in_population = len(control_videos)

    titles = ["CP", "BD"]
    b_o_colors = ['#1f77b4', '#ff7f0e']

    t_max = (15 * 60 * 30) - window_size  # ds["start_frame"].max()
#%% plot average metric per population
def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)
def error_bar(arrs, axis=0):
    arrs = np.array(arrs)
    return np.nanmean(arrs, axis=axis), np.nanstd(arrs, axis=axis, ddof=1)/np.sqrt(np.size(arrs, axis=axis))

num_metrics = 5
metric_names = [#"entropy",
                #"effect_num_states"
                #"num_zero_row",
                #"num_one_item",
                #"num_zero_item",
                "latent_volume_all_motifs" ,
                "start_frame",
                "is_BD"
                ]
lims = [[-0.5, 2.3], [-5, 15], [-4, 8], [30, 120]]
CP_idx = np.zeros(n_subject_in_population)
BD_idx = np.ones(n_subject_in_population)
sns.set_style('white')
for i in range(num_metrics):
    fig, axes = plt.subplots(1, figsize=(10, 5))
    for group in range(2):
        ds1 = ds[ds["is_BD"] == group]
        metric_mean_over_sub = []
        metric_ste_over_sub = []
        for t in range(t_max):
            ds_t = ds1[ds1["start_frame"] == t]
            y, error = error_bar(ds_t[metric_names[i]], axis=0)
            metric_mean_over_sub.append(ds_t[metric_names[i]].mean())
            metric_ste_over_sub.append(ds_t[metric_names[i]].std()/np.sqrt(len(ds_t[metric_names[i]])))
        x = np.arange(t_max)
        metric_mean_over_sub = np.asarray(metric_mean_over_sub)
        metric_ste_over_sub = np.asarray(metric_ste_over_sub)
        line = axes.plot(x, metric_mean_over_sub, color=b_o_colors[group].format(group), zorder=1)


        axes.fill_between(x, metric_mean_over_sub - metric_ste_over_sub, metric_mean_over_sub + metric_ste_over_sub, norm=plt.Normalize(vmin=0, vmax=9),
                         alpha=0.2, facecolor=b_o_colors[group].format(group))

        axes.set_title('average {}'.format(metric_names[i]))

        axes.set_xlim([0, t_max])
        axes.set_xlabel('population')
    sns.despine()
    plt.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\transition_metrics'.format(onedrive_path, project_name)
    Path.mkdir(Path(pwd), exist_ok=True)
    fname = 'average {}-{}motifs_new.png'.format(metric_names[i], n_cluster)
    fname_pdf = 'average {}-{}motifs_new.pdf'.format(metric_names[i], n_cluster)
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)


#plt.suptitle("sliding window")

#%% Plot per patient L0 metrics
num_metrics = 1
CP_idx = np.zeros(n_subject_in_population)
BD_idx = np.ones(n_subject_in_population)
color = sns.color_palette("tab10")
for i in range(num_metrics):
    for j, videos in enumerate([control_videos, BD_videos]):
        for sub in range(n_subject_in_population):
            fig, axes = plt.subplots(1, figsize=(10, 5))
            sns.set_style('white')
            if sub == 1 or sub == n_subject_in_population:
                leg = True
            else:
                leg = False
            sub_name = videos[sub]
            ds1 = ds[ds["subject"] == sub_name]
            line = sns.scatterplot(data=ds1, x="start_frame", y=metric_names[i],
                                hue="is_BD",  ax=axes, legend=leg,
                                linewidth=0, alpha=1, s=5, palette={0:color[0], 1:color[1]})#palette={0:'C0', 1:'C1'},
            sns.despine()
            axes.set_ylim(lims[i])
            axes.set_xlim([0, t_max])
            axes.set_title('subject {}'.format(sub_name))
            axes.set_xlabel('population')
            #axes.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.show()
            pwd = r'{}\Behavior_VAE_data\{}\figure\transition_metrics'.format(onedrive_path, project_name)
            fname = "{}_{}.png".format(metric_names[i], sub_name)
            fig.savefig(os.path.join(pwd, fname), transparent=True)
            fname1 = "{}_{}.pdf".format(metric_names[i], sub_name)
            fig.savefig(os.path.join(pwd, fname1), transparent=True)

#%% First half second half analysis



#%% Per patient, entropy, and latent volume per motif second half/ first half
n_latent = 10
groups = ['CP', 'BD']
entropy_df_first_second = []
latent_volume_df_first_second = []
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

latent_volume0_dif = []
latent_volume1_dif = []
latent_volume2_dif = []
latent_volume3_dif = []
latent_volume4_dif = []
latent_volume5_dif = []
latent_volume6_dif = []
latent_volume7_dif = []
latent_volume8_dif = []
latent_volume9_dif = []
is_BD = []
latent_d = []
for j, videos in enumerate([control_videos, BD_videos]):
    for sub in range(n_subject_in_population):
        sub_name = videos[sub]
        df1 = ds[ds["subject"] == sub_name]
        total_len = t_max
        df2 = df1["entropy"]
        entropy_first_half = np.nanmean(df2[:total_len//2])
        entropy_second_half = np.nanmean(df2[total_len //2:])

        entropy_df_first_second.append(np.abs(entropy_second_half-entropy_first_half))
        is_BD.append(j)
        for d in range(n_cluster):
            y = df1['latent_volume_motif{}'.format(d)].to_numpy()
            latent_v_first_half = np.nanmean(y[:total_len // 2])
            latent_v_second_half = np.nanmean(y[total_len // 2:])
            eval('latent_volume{}_dif'.format(d)).append(latent_v_second_half - latent_v_first_half)
            eval('latent_volume{}'.format(d)).append(y[:t_max])
        latent_volume_df_first_second.append(eval('latent_volume{}'.format(d)))
entropy_df = pd.DataFrame(np.asarray([entropy_df_first_second, is_BD]).T, columns=['metric','is_BD'])
#%% Plot entropy and latent volume change in first half and second half
fig, ax = plt.subplots(1, figsize=(10, 5))
sns.boxplot(y="metric", x='is_BD', hue='is_BD', data=entropy_df, orient="v",palette=sns.color_palette("tab10"))
ax.set_xticklabels(['CP','BD'])
ax.set_title('change in entropy')
ax.set_ylabel('second half - first half ')
fig.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\transition_metrics'.format(onedrive_path, project_name)
fname = "entropy_first_over_second.png"
fig.savefig(os.path.join(pwd, fname), transparent=True)
fname1 = "entropy_first_over_second.pdf"
fig.savefig(os.path.join(pwd, fname1), transparent=True)

latent_df = pd.DataFrame(np.asarray([latent_volume0, latent_volume1, latent_volume2, latent_volume3, latent_volume4,
                                     latent_volume5, latent_volume6, latent_volume7, latent_volume8, latent_volume9,
                                     is_BD]).T,
                         columns=['latent_volume0', 'latent_volume1', 'latent_volume2', 'latent_volume3', 'latent_volume4',
                                  'latent_volume5', 'latent_volume6', 'latent_volume7', 'latent_volume8', 'latent_volume9','is_BD'])

for d in range(n_cluster):
    fig, ax = plt.subplots(1, figsize=(10, 5))
    sns.boxplot(y="latent_volume{}".format(d), x='is_BD', hue='is_BD', data=latent_df, orient="v",palette=sns.color_palette("tab10"))
    ax.set_xticklabels(['CP', 'BD'])
    ax.set_ylabel('second half  - first half ')
    ax.set_title('change in latent volume motif {}'.format(d))
    ax.set_ylim([-900, 600])
    fig.show()
    pwd = r'{}\Behavior_VAE_data\BD20-Jun5-2022\figure\latent_slide_window'
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "latent_volume_motif_{}_first_over_second.png".format(d)
    fig.savefig(os.path.join(pwd, fname))
    fname1 = "latent_volume_motif_{}_first_over_second.pdf".format(d)
    fig.savefig(os.path.join(pwd, fname1), transparent=True)
#%% entropy diff stat tests
from scipy import stats
CP = entropy_df['metric'][:n_subject_in_population].to_numpy()
BD = entropy_df['metric'][n_subject_in_population:].to_numpy()
res = stats.ttest_ind(CP, BD)
print(res.pvalue)
f= stats.f_oneway(CP, BD)

F = np.var(CP) / np.var(BD)
df1 = len(CP) - 1
df2 = len(BD) - 1
alpha = 0.05 #Or whatever you want your alpha to be.
p_value = stats.f.cdf(F, df1, df2)
print(p_value)
#%% volume diff stat tests
for d in range(n_latent):
    print("latent volume  {} ".format(d))

    latent_df = pd.DataFrame(np.asarray([eval('latent_volume{}_dif'.format(d)), is_BD]).T, columns=['metric','is_BD'])
    latent_motif = eval('latent_volume{}'.format(d))
    CP = latent_df['metric'][:n_subject_in_population].to_numpy()
    BD = latent_df['metric'][n_subject_in_population:].to_numpy()

    CP_vol = np.asarray(latent_motif[:n_subject_in_population])
    BD_vol = np.asarray(latent_motif[n_subject_in_population:])
    #TODO: check what stat to use to test 25 observations of two distributions
    #TODO: check some volume is nan or zero
    res = stats.ttest_ind(CP_vol, BD_vol)

    f = stats.f_oneway(CP, BD)
    F = np.var(CP) / np.var(BD)
    df1 = len(CP) - 1
    df2 = len(BD) - 1
    alpha = 0.05  # Or whatever you want your alpha to be.
    p_value = stats.f.cdf(F, df1, df2)
    print(" F test for sec - first pvalue {}".format(p_value))
    print(" t test for BD-HP pvalue {}".format(res.pvalue))
#%% Plot per patient change of dwell time, and latent volume
from itertools import zip_longest
latent_d = 10
n_cluster = 10
CP_idx = np.zeros(n_subject_in_population)
BD_idx = np.ones(n_subject_in_population)
cmap = plt.get_cmap('tab20')
lims = [[-500, 2000],[-0.2, 1.2]]
groups = ['Control', 'BD']
CP_mean_motifs = [[],[]]
BD_mean_motifs = [[],[]]

latent_length = 27000 - window_size

for j, videos in enumerate([control_videos, BD_videos]):
    group = groups[j]
    mean_motif_freq = [[] for _ in range(n_cluster)]
    mean_motif_volume = [[] for _ in range(n_cluster)]
    for sub in range(n_subject_in_population):
        fig, ax = plt.subplots(1, figsize=(10, 5))
        fig1, ax1 = plt.subplots(1, figsize=(10, 5))
        sub_name = videos[sub]
        ds1 = ds[ds["subject"] == sub_name]
        for d in range(n_cluster):
            x = ds1["start_frame"].to_numpy()
            y = ds1['latent_volume_motif{}'.format(d)].to_numpy()
            z = ds1['motif{}_usage_freq'.format(d)].to_numpy()
            mean_motif_freq[d].append(z[: latent_length])
            mean_motif_volume[d].append(y[: latent_length])
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

        pwd = r'{}\Behavior_VAE_data\{}\figure\latent_slide_window'.format(onedrive_path, project_name)
        Path(pwd).mkdir(parents=True, exist_ok=True)
        fname = "{}_{}.png".format('latent_colume', sub_name)
        #fig.savefig(os.path.join(pwd, fname), transparent=True)
        fname0 = "{}_{}.pdf".format('latent_colume', sub_name)
        #fig.savefig(os.path.join(pwd, fname0), transparent=True)

        pwd = r'{}\Behavior_VAE_data\{}\figure\motif_freq_slide_window'.format(onedrive_path, project_name)
        Path(pwd).mkdir(parents=True, exist_ok=True)
        fname = "{}_{}.png".format('motif_usage', sub_name)
        #fig1.savefig(os.path.join(pwd, fname), transparent=True)
        fname1 = "{}_{}.pdf".format('motif_usage', sub_name)
        #fig1.savefig(os.path.join(pwd, fname1), transparent=True)
    if j == 0:
        CP_mean_motifs = [mean_motif_freq, mean_motif_volume]
    if j == 1:
        BD_mean_motifs = [mean_motif_freq, mean_motif_volume]
plt.close('all')
#%% Plot per population change of dwell time, and latent volume
from scipy.stats import pearsonr
latent_d = 10
n_cluster = 10
CP_idx = np.zeros(n_subject_in_population)
BD_idx = np.ones(n_subject_in_population)
cmap = plt.get_cmap('tab20')
lims = [[-100, 800],[-0.1, 0.4]]
groups = ['CP', 'BD']


for d in range(n_cluster):
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig1, ax1 = plt.subplots(1, figsize=(10, 5))
    for j, videos in enumerate([control_videos, BD_videos]):
        group = groups[j]
        mean_motif_freq = CP_mean_motifs[0][d] if j == 0 else BD_mean_motifs[0][d]
        z, error_z = error_bar(mean_motif_freq)

        mean_motif_volume = CP_mean_motifs[1][d] if j == 0 else BD_mean_motifs[1][d]
        y, error_y = error_bar(mean_motif_volume)

        r = [np.corrcoef(mean_motif_freq[i], mean_motif_volume[i])[1, 0] for i in range(len(videos))]

        print(f'{group} motif {d} corr r:{np.nanmean(r)}+- {np.nanstd(r)}')

        x = np.arange(len(y)) + 1
        ax.fill_between(x, y - error_y, y + error_y, norm=plt.Normalize(vmin=0, vmax=9),
                        alpha=0.2, facecolor=cmap(d * 2 + j))
        ax.plot(x, y, color=cmap(d * 2 + j), label='{}-{}'.format(group, d), zorder=1)

        x1 = np.arange(len(z)) + 1
        ax1.fill_between(x1, z - error_z, z + error_z, norm=plt.Normalize(vmin=0, vmax=9),
                         alpha=0.2, facecolor=cmap(d * 2 + j))
        ax1.plot(x1, z, color=cmap(d * 2 + j), label='{}-{}'.format(group, d), zorder=1)

    ax.axhline(0, color='k',linestyle="dashed")
    ax.set_ylim(lims[0])
    ax.set_xlim([0,t_max])
    ax.set_title('{}-{}-latent volume motif {} +- sem'.format('BD-CP', 'average', d))
    ax.set_xlabel('time (frames)')

    ax1.axhline(0, color='white',linestyle="dashed")
    ax1.set_ylim(lims[1])
    ax1.set_xlim([0, t_max])
    ax1.set_title('{}-{}-motif{} frequency'.format('BD-CP', 'average', d))
    ax1.set_xlabel('time (frames)')

    ax.legend()
    ax1.legend()
    ax.grid(False)
    ax1.grid(False)

    fig.show()

    pwd = r'{}\Behavior_VAE_data\{}\figure\latent_slide_window'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "{}_{}_motif{}-new.png".format('latent_volume', 'BD-CP', d)
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fname0 = "{}_{}_motif{}-new.pdf".format('latent_volume', 'BD-CP', d)
    fig.savefig(os.path.join(pwd, fname0), transparent=True)

    fig1.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\motif_freq_slide_window'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "{}_{}_motif{}.png".format('motif_usage', 'BD-CP', d)
    fig1.savefig(os.path.join(pwd, fname), transparent=True)
    fname1 = "{}_{}_motif{}.pdf".format('motif_usage', 'BD-CP', d)
    fig1.savefig(os.path.join(pwd, fname1), transparent=True)
#%%
# num_metrics = 5
# CP_idx = np.zeros(n_subject_in_population)
# BD_idx = np.ones(n_subject_in_population)
#
#
# for i in range(num_metrics):
#     fig, axes = plt.subplots(1, figsize=(10, 5))
#     sns.set_style('darkgrid')
#     for j, videos in enumerate([control_videos, BD_videos]):
#         for sub in range(n_subject_in_population):
#             if sub == 1 or sub == n_subject_in_population:
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