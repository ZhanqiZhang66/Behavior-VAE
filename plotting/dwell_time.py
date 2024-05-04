# Created by Victoria Zhang at 8/14/2022
# File: dwell_time.py
# Description: load dwell time per state and analyze the distribution over 4 time episodes.
# Scenario:
# Usage:
#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy import stats
# from scipy.stats import permutation_test
from pathlib import Path
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
from data.load_data import load_pt_data
from plotting.get_paths import get_my_path
#%%
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']
#%%
b_o_colors = ['#1f77b4', '#DE3163']
#%%
project_name = 'BD25-HC25-final-May17-2023'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(cfg['project_path'],"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
n_scores = 11
model_name = 'VAME'
five_min_frame_no = int(5 * 60 * 30)

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
score_bahavior_names = ["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
n_subject_in_population = len(control_videos)
#%%
YMRS_score = []
HAM_D_score = []
titles = ["HC", "BD"]
matching_labels = 0
total_label_chunks = 0

Motif_usages = [[], []]
Motif_usage_pct = [[], []]
motif_usage_cat = [[], []]
Labels = [[], []]
Cluster_center = []

Motif_usages_ctl = [[], []]
Motif_usage_pct_ctl = [[], []]
motif_usage_cat_ctl = [[], []]
Labels_ctl = [[], []]
Cluster_center_ctl = []
Labels_score = [[], []]
motif_usage_cat_score = [[], []]

Motif_usages_score = [[], []]
Motif_usage_pct_score = [[], []]

Epoch1_labels = [[], []]
Epoch1_motif_usage = [[], []]
Epoch1_motif_usage_cat = [[],[]]

Epoch2_labels = [[], []]
Epoch2_motif_usage = [[], []]
Epoch2_motif_usage_cat = [[],[]]

Epoch3_labels = [[], []]
Epoch3_motif_usage = [[], []]
Epoch3_motif_usage_cat = [[],[]]

Epoch1_labels_ctl = [[], []]
Epoch1_motif_usage_ctl = [[], []]
Epoch1_motif_usage_cat_ctl = [[],[]]

Epoch2_labels_ctl = [[], []]
Epoch2_motif_usage_ctl = [[], []]
Epoch2_motif_usage_cat_ctl = [[],[]]

Epoch3_labels_ctl = [[], []]
Epoch3_motif_usage_ctl = [[], []]
Epoch3_motif_usage_cat_ctl = [[],[]]

Epoch1_labels_score = [[], []]
Epoch1_motif_usage_score = [[], []]
Epoch1_motif_usage_cat_score = [[],[]]

Epoch2_labels_score = [[], []]
Epoch2_motif_usage_score = [[], []]
Epoch2_motif_usage_cat_score = [[],[]]

Epoch3_labels_score = [[], []]
Epoch3_motif_usage_score = [[], []]
Epoch3_motif_usage_cat_score = [[],[]]
def get_motif_usage(label, n_cluster):
    motif_usage = np.asarray([label.tolist().count(i) for i in range(n_cluster)])
    return motif_usage
def divide_with_delimiter(lst):
    # Example usage:
    #X = [1, 1, 1, 2, 2, 3, 3, 3]
    # result =  [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    #
    chunks = []
    for i, num in enumerate(lst):
        if i == 0 or num != lst[i - 1]:
            chunks.append([num])
        else:
            chunks[-1].append(num)

    result_length = sum(len(chunk) + 1 for chunk in chunks) - 1
    result = [0] * result_length
    index = 0
    for chunk in chunks:
        result[index:index + len(chunk)] = [1] * len(chunk)
        index += len(chunk) + 1

    return result

def count_closest_ones(list1, list2, offset=0):
    # Example usage:
    # list1 = [0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    # list2 = [1, 0, 1, 1, 1, 0, 1, 1, 1, 0]
    # x = 2
    # return 2
    count = 0
    for i, val1 in enumerate(list1):
        if val1 == 0:
            closest_distance = float('inf')
            for j, val2 in enumerate(list2):
                if val2 == 0:
                    distance = abs(j - i)
                    if distance <= offset:
                        closest_distance = min(closest_distance, distance)
            if closest_distance <= offset:
                count += 1
    return count




for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        YMRS_score.append(YMRS[v])
        HAM_D_score.append(HAM_D[v])
        print("Loading {}-{} data {}/{}...".format(v, titles[j], i+1, len(videos)))
        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        motif_usage = get_motif_usage(label, n_cluster)
        np.save(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v), motif_usage)
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")

        if n_cluster == 10:
            control_label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
            control_cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, n_cluster, v))
            control_motif_usage = get_motif_usage(control_label, n_cluster)

            score_label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\score_labels_{}.npy'.format(onedrive_path, project_name, v,n_cluster,v))
            # score_label = score_label[:15*60*30]
            score_label_merged = score_label.copy()
            score_motif_usage = get_motif_usage(score_label, n_scores)

        label_delimiter = divide_with_delimiter(label[:five_min_frame_no*3])
        score_label_delimiter = divide_with_delimiter(score_label[:five_min_frame_no*3])
        total_label_chunks += score_label_delimiter.count(0)
        matching_chunks = count_closest_ones(divide_with_delimiter(score_label[:five_min_frame_no*3]),
                                                 divide_with_delimiter(label[:five_min_frame_no*3]),
                                                 offset=30)
        print(f"{v} has {matching_chunks} matching labels between score and VAME labels")
        matching_labels += matching_chunks
        # bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
        # score_label_merged[score_label_merged == 1] = 0 #sit
        #
        # score_label_merged[score_label_merged == 2] = 1 #stand
        # score_label_merged[score_label_merged == 3] = 1 #stand
        #
        # score_label_merged[score_label_merged == 4] = 2 #walk
        # score_label_merged[score_label_merged == 5] = 2 #walk
        #
        # score_label_merged[score_label_merged == 6] = 3 #lie
        # score_label_merged[score_label_merged == 7] = 3 #lie
        # score_label_merged[score_label_merged == 8] = 4  # interact
        # score_label_merged[score_label_merged == 9] = 5  # wear



        if n_cluster == 10:

            # ----------DLC control----------concat labels, motif usage for 3 epochs for control
            epoch_1_label_ctl = control_label[:five_min_frame_no + offset]
            epoch_2_label_ctl = control_label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
            epoch_3_label_ctl = control_label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

            epoch_1_motif_usage_ctl = get_motif_usage(epoch_1_label_ctl, n_cluster)
            epoch_2_motif_usage_ctl = get_motif_usage(epoch_2_label_ctl, n_cluster)
            epoch_3_motif_usage_ctl = get_motif_usage(epoch_3_label_ctl, n_cluster)

            Epoch1_labels_ctl[j].append(epoch_1_label_ctl)
            Epoch1_motif_usage_ctl[j].append(epoch_1_motif_usage_ctl / np.sum(epoch_1_motif_usage_ctl))

            Epoch2_labels_ctl[j].append(epoch_2_label_ctl)
            Epoch2_motif_usage_ctl[j].append(epoch_2_motif_usage_ctl / np.sum(epoch_2_motif_usage_ctl))

            Epoch3_labels_ctl[j].append(epoch_3_label_ctl)
            Epoch3_motif_usage_ctl[j].append(epoch_3_motif_usage_ctl / np.sum(epoch_3_motif_usage_ctl))

            # ----------SCORES-----------concat labels, motif usage for 3 epochs for scores--------
            epoch_1_label_score = score_label_merged[:five_min_frame_no + offset]
            epoch_2_label_score = score_label_merged[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
            epoch_3_label_score = score_label_merged[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

            epoch_1_motif_usage_score = get_motif_usage(epoch_1_label_score, n_scores)
            epoch_2_motif_usage_score = get_motif_usage(epoch_2_label_score, n_scores)
            epoch_3_motif_usage_score = get_motif_usage(epoch_3_label_score, n_scores)

            Epoch1_labels_score[j].append(epoch_1_label_score)
            Epoch1_motif_usage_score[j].append(epoch_1_motif_usage_score / np.sum(epoch_1_motif_usage_score))

            Epoch2_labels_score[j].append(epoch_2_label_score)
            Epoch2_motif_usage_score[j].append(epoch_2_motif_usage_score / np.sum(epoch_2_motif_usage_score))

            Epoch3_labels_score[j].append(epoch_3_label_score)
            Epoch3_motif_usage_score[j].append(epoch_3_motif_usage_score / np.sum(epoch_3_motif_usage_score))

        door_close_time = start_frame[v]
        start_time = start_frame[v] #start_frame.loc[v_index, 'n']

        offset = 0 #int(start_time - door_close_time)

        #---------OURS--------- concat labels, motif usage for 3 epochs
        epoch_1_label = label[:five_min_frame_no + offset]
        epoch_2_label = label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label = label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

        epoch_1_motif_usage = get_motif_usage(epoch_1_label, n_cluster)
        epoch_2_motif_usage = get_motif_usage(epoch_2_label, n_cluster)
        epoch_3_motif_usage = get_motif_usage(epoch_3_label, n_cluster)

        Epoch1_labels[j].append(epoch_1_label)
        Epoch1_motif_usage[j].append(epoch_1_motif_usage/ np.sum(epoch_1_motif_usage))

        Epoch2_labels[j].append(epoch_2_label)
        Epoch2_motif_usage[j].append(epoch_2_motif_usage/ np.sum(epoch_2_motif_usage))

        Epoch3_labels[j].append(epoch_3_label)
        Epoch3_motif_usage[j].append(epoch_3_motif_usage/ np.sum(epoch_3_motif_usage))




        if i == 0:
            l = label


            m = motif_usage


            Cluster_center.append(cluster_center)
            m_e1 = epoch_1_motif_usage
            m_e2 = epoch_2_motif_usage
            m_e3 = epoch_3_motif_usage


            if n_cluster == 10:
                l_control = control_label
                l_score = score_label
                m_control = control_motif_usage
                m_score = score_motif_usage
                Cluster_center_ctl.append(control_cluster_center)
                m_e1_ctl = epoch_1_motif_usage_ctl
                m_e2_ctl = epoch_2_motif_usage_ctl
                m_e3_ctl = epoch_3_motif_usage_ctl

                m_e1_score = epoch_1_motif_usage_score
                m_e2_score = epoch_2_motif_usage_score
                m_e3_score = epoch_3_motif_usage_score
            # print(np.shape(m_e3))
        else:
            l = np.concatenate([l,label])
            m += motif_usage
            Cluster_center.append(cluster_center)
            m_e1 += np.asarray(epoch_1_motif_usage)
            m_e2 += np.asarray(epoch_2_motif_usage)
            m_e3 += np.asarray(epoch_3_motif_usage)


            if n_cluster == 10:
                l_control = np.concatenate([l_control, control_label])
                m_control += control_motif_usage
                Cluster_center_ctl.append(control_cluster_center)
                m_e1_ctl += np.asarray(epoch_1_motif_usage_ctl)
                m_e1_ctl += np.asarray(epoch_2_motif_usage_ctl)
                m_e1_ctl += np.asarray(epoch_3_motif_usage_ctl)

                l_score = np.concatenate([l_score, score_label])
                m_score += score_motif_usage
                m_e1_score += np.asarray(epoch_1_motif_usage_score)
                m_e2_score += np.asarray(epoch_2_motif_usage_score)
                m_e3_score += np.asarray(epoch_3_motif_usage_score)
            # print(np.shape(m_e3))

        motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
        if n_cluster == 10:
            motif_usage_cat_ctl[j].append(control_motif_usage / np.sum(control_motif_usage))
            motif_usage_cat_score[j].append(score_motif_usage / np.sum(score_motif_usage))

        num_points = label.shape[0]
        n += num_points

    Epoch1_motif_usage_cat[j] = m_e1
    Epoch2_motif_usage_cat[j] = m_e2
    Epoch3_motif_usage_cat[j] = m_e3
    Motif_usages[j] = m
    Motif_usage_pct[j] = m/n
    Labels[j] = l
    if n_cluster == 10:
        Epoch1_motif_usage_cat_ctl[j] = m_e1_ctl
        Epoch2_motif_usage_cat_ctl[j] = m_e2_ctl
        Epoch3_motif_usage_cat_ctl[j] = m_e3_ctl
        Motif_usages_ctl[j] = m_control
        Motif_usage_pct_ctl[j] = m_control / n
        Labels_ctl[j] = l_control

        Epoch1_motif_usage_cat_score[j] = m_e1_score
        Epoch2_motif_usage_cat_score[j] = m_e2_score
        Epoch3_motif_usage_cat_score[j] = m_e3_score
        Motif_usages_score[j] = m_score
        Motif_usage_pct_score[j] = m_score / n
        Labels_score[j] = l_control


print(f"{matching_labels/total_label_chunks} labels are matched between score and VAME labels")
#%% Population-wise analysis
transition_group = ['','_ctl', '_score']
motif_usage_cat = np.asarray(motif_usage_cat)
motif_usage_cat_ctl = np.asarray(motif_usage_cat_ctl)
motif_usage_cat_score = np.asarray(motif_usage_cat_score)
diagnostic_path = rf"{data_path}\scaled_diagnostic_data.csv"
assessment_df = pd.read_csv(diagnostic_path)
assessment_df.drop('gender', axis=1, inplace=True)

bd_df = assessment_df[['video', 'BD']]
motif_usage_cat_s3d = pd.read_csv(rf'{data_path}\S3D\motif_usage_overall.csv')
motif_usage_cat_s3d = pd.merge(bd_df, motif_usage_cat_s3d, on='video')
#%% between motif paired t test and score correlation
import permutation_test as p
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i in range(n_cluster):
    HC = motif_usage_cat[0,:,i].reshape(-1,1)
    BD = motif_usage_cat[1,:,i].reshape(-1,1)
    s = stats.ttest_ind(HC, BD)
    # because our statistic is vectorized, we pass `vectorized=True`
    # `n_resamples=np.inf` indicates that an exact test is to be performed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
    # res = p.permutation_test(BD, HC)
    print("Motif {}".format(i))
    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s.statistic[0], s.pvalue[0]))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score = scipy.stats.pearsonr(motif_usage_cat[0,:,i], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score = scipy.stats.pearsonr(motif_usage_cat[0,:,i], YMRS_score[:n_subject_in_population])
    print("          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
    print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score[0][0], corr_HAM_D_score[1]))

    # only correlate with BD list
    corr_HAM_D_score_BD = scipy.stats.pearsonr(motif_usage_cat[1,:,i], HAM_D_score[n_subject_in_population:])
    corr_YMRS_score_BD = scipy.stats.pearsonr(motif_usage_cat[1,:,i], YMRS_score[n_subject_in_population:])
    print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0][0], corr_YMRS_score_BD[1]))
    print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_BD[0][0], corr_HAM_D_score_BD[1]))

    HC_ctl = motif_usage_cat_ctl[0,:,i].reshape(-1,1)
    BD_ctl = motif_usage_cat_ctl[1,:,i].reshape(-1,1)
    s_ctl = stats.ttest_ind(HC_ctl, BD_ctl)
    # because our statistic is vectorized, we pass `vectorized=True`
    # `n_resamples=np.inf` indicates that an exact test is to be performed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
    # res = p.permutation_test(BD, HC)
    print(" Control \n")
    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s_ctl.statistic[0], s_ctl.pvalue[0]))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[0,:,i], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[0,:,i], YMRS_score[:n_subject_in_population])
    print("          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_ctl[0][0], corr_YMRS_score_ctl[1]))
    print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_ctl[0][0], corr_HAM_D_score_ctl[1]))
    # only correlate with BD list
    corr_HAM_D_score_BD_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[1,:,i], HAM_D_score[n_subject_in_population:])
    corr_YMRS_score_BD_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[1,:,i], YMRS_score[n_subject_in_population:])
    print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD_ctl[0][0], corr_YMRS_score_BD_ctl[1]))
    print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_BD_ctl[0][0], corr_HAM_D_score_BD_ctl[1]))

#%% Scored motif between motif paired t test and score correlation
motif_usage_cat_score = np.asarray(motif_usage_cat_score)
for i, motif_behavior in enumerate(score_bahavior_names):
    HC_score = motif_usage_cat_score[0, :, i].reshape(-1, 1)
    BD_score = motif_usage_cat_score[1, :, i].reshape(-1, 1)
    print("Score motif  {}\n".format(motif_behavior))
    if np.sum(HC_score) == 0 or np.sum(BD_score) == 0:
        print("HC score of this class does not exist ", np.sum(HC_score) == 0)
        print("BD score of this class does not exist ", np.sum(BD_score) == 0)
        continue
    s_score = stats.ttest_ind(HC_score, BD_score, nan_policy='omit')

    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s_score.statistic[0], s_score.pvalue[0]))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score_score = scipy.stats.pearsonr(motif_usage_cat_score[0, :, i], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score_score = scipy.stats.pearsonr(motif_usage_cat_score[0, :, i], YMRS_score[:n_subject_in_population])
    print("          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_score[0][0],
                                                                               corr_YMRS_score_score[1]))
    print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_score[0][0],
                                                                               corr_HAM_D_score_score[1]))
    # only correlate with BD list
    corr_HAM_D_score_BD_score = scipy.stats.pearsonr(motif_usage_cat_score[1, :, i], HAM_D_score[n_subject_in_population:])
    corr_YMRS_score_BD_score = scipy.stats.pearsonr(motif_usage_cat_score[1, :, i], YMRS_score[n_subject_in_population:])
    print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD_score[0][0],
                                                                               corr_YMRS_score_BD_score[1]))
    print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD_score[0][0],
                                                                               corr_HAM_D_score_BD_score[1]))


#%% Plot Box
bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear"]
states = []
for i in range(n_cluster):
    states.append([i]*n_subject_in_population)
states = np.asarray(states).flatten()
sns.set_style("white")
for k in range(3):
    HC_idx = np.zeros(n_subject_in_population * n_cluster)
    BD_idx = np.ones(n_subject_in_population * n_cluster)
    motif_usage_to_plot = np.array(eval("motif_usage_cat{}".format(transition_group[k])))
    motif_usage_to_plot_ = np.zeros((2, n_subject_in_population, n_cluster))
    motif_usage_to_plot_ = motif_usage_to_plot[:,:,:n_cluster]
    ds = pd.DataFrame(np.concatenate((
        np.concatenate((motif_usage_to_plot_[0,:,:].T.flatten(), motif_usage_to_plot_[1,:,:].T.flatten()), 0).reshape(-1, 1),
        np.concatenate((HC_idx, BD_idx), 0).reshape(-1, 1),
        np.concatenate((states, states), 0).reshape(-1, 1)), 1),
        columns=['motif frequency','is_BD','state'])
    w = n_cluster/10 * 6
    fig, ax = plt.subplots(1, 1, figsize=(w, 4))
    violin = sns.violinplot(y="motif frequency", x='state',hue='is_BD',
                   data=ds, orient="v", palette=sns.color_palette("tab10"),split=True)
    handles = violin.legend_.legendHandles
    dict_name = {0.0:'HC', 1.0:'BD'}
    labels = [dict_name[float(text.get_text())] for text in ax.legend_.texts]
    # sns.swarmplot(y="motif frequency", x="state", hue='is_BD',data=ds,dodge=True,size=2)
    x = np.arange(n_cluster)
    ax.legend(handles, labels)
    ax.set_xticks(x)
    if k == 2:
        ax.set_xticklabels(bahavior_names)

    ax.set_title('15 min dwell frequency over {} motifs {}'.format(n_cluster, transition_group[k]))
    ax.set_xlabel('Motifs(States)')
    sns.despine()
    fig.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "15-min-dwell-{}.png".format(transition_group[k])
    fname_pdf = "15-min-dwell-{}.pdf".format(transition_group[k])
    # fig.savefig(os.path.join(pwd, fname), transparent=True)
    # fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

#%% merge state for human labels
bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear"]
merged_states = []
for i in range(6):
    merged_states.append([i]*n_subject_in_population)
merged_states = np.asarray(merged_states).flatten()
HC_idx = np.zeros(n_subject_in_population * 6)
BD_idx = np.ones(n_subject_in_population * 6)
motif_usage_to_plot = np.array(eval("motif_usage_cat{}".format(transition_group[2])))
motif_usage_to_plot_ = np.zeros((2, n_subject_in_population, 6))
motif_usage_to_plot_[:,:,0] = motif_usage_to_plot[:,:,0] + motif_usage_to_plot[:,:,1]
motif_usage_to_plot_[:,:,1] = motif_usage_to_plot[:,:,2] + motif_usage_to_plot[:,:,3]
motif_usage_to_plot_[:,:,2] = motif_usage_to_plot[:,:,4] + motif_usage_to_plot[:,:,5]
motif_usage_to_plot_[:,:,3] = motif_usage_to_plot[:,:,6] + motif_usage_to_plot[:,:,7]
motif_usage_to_plot_[:,:,4] = motif_usage_to_plot[:,:,8]
motif_usage_to_plot_[:,:,5] = motif_usage_to_plot[:,:,9]
ds = pd.DataFrame(np.concatenate((
    np.concatenate((motif_usage_to_plot_[0,:,:].T.flatten(), motif_usage_to_plot_[1,:,:].T.flatten()), 0).reshape(-1, 1),
    np.concatenate((HC_idx, BD_idx), 0).reshape(-1, 1),
    np.concatenate((merged_states, merged_states), 0).reshape(-1, 1)), 1),
    columns=['motif frequency','is_BD','state'])
w = 6/10 * 6
fig, ax = plt.subplots(1, 1, figsize=(w, 4))
violin = sns.boxplot(y="motif frequency", x='state', hue='is_BD',
                     data=ds, orient="v", palette=sns.color_palette("tab10"))
handles = violin.legend_.legendHandles
dict_name = {0.0: 'HP', 1.0: 'BD'}
labels = [dict_name[float(text.get_text())] for text in ax.legend_.texts]
# sns.swarmplot(y="motif frequency", x="state", hue='is_BD',data=ds,dodge=True,size=2)
x = np.arange(6)
ax.legend(handles, labels)
ax.set_xticks(x)
ax.set_xticklabels(["sit", "stand",  "walk", "lie", "interact", "wear"])
fig.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
Path(pwd).mkdir(parents=True, exist_ok=True)
fname = "15-min-dwell-merged-score.png"
fname_pdf = "15-min-dwell-merged-score.pdf"
fig.savefig(os.path.join(pwd, fname), transparent=True)
fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%% plot box dwell per video
sns.set_style('white')

states = []
for i in range(n_cluster):
    states.append([i])
states = np.asarray(states).flatten()

for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for p in range(len(videos)):
        v = videos[p]

        idx = np.ones(n_cluster) * j

        ds = pd.DataFrame(np.concatenate((
            motif_usage_cat[j,p,:].T.flatten().reshape(-1, 1),
            idx.reshape(-1, 1),
            states.reshape(-1, 1)), 1),
            columns=['motif frequency','is_BD','state'])
        w = n_cluster/10 * 6
        fig, ax = plt.subplots(1, 1, figsize=(w, 4))
        violin = sns.barplot(y="motif frequency", x='state',hue='is_BD',
                       data=ds, orient="v", facecolor=b_o_colors[j])
        x = np.arange(n_cluster)
        ax.set_xticks(x)
        ax.set_title('{} 15 min dwell frequency over {} motifs'.format(v, n_cluster))
        ax.set_xlabel('Motifs(States)')
        ax.set_ylim([0, 1])
        sns.despine()
        fig.show()
        pwd = r'{}\Behavior_VAE_data\{}\figure\motif_freq_each_video'.format(onedrive_path, project_name)
        Path(pwd).mkdir(parents=True, exist_ok=True)
        fname = "{}_{}_motif_freq.png".format(n_cluster, v)
        fig.savefig(os.path.join(pwd, fname))
#%% Plot histogram of averaged dwell frequency
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
from scipy import stats
from scipy.stats import ks_2samp

p = stats.ttest_ind(Motif_usages[0]/N[0], Motif_usages[1]/N[1])
x = np.arange(n_cluster)
for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    motif = Motif_usages[j]
    label = Labels[j]
    tm = TM[j]
    ax.bar(x + j* 0.4, motif / np.sum(motif), width=0.4)
ax.set_xlabel("Motifs(States)")
ax.set_ylabel("dwell frequency")
ax.set_xticks(x)
ax.set_title('average dwell frequency over {} motifs'.format(n_cluster))
ax.set_ylim([0, 0.2])
ax.legend(titles)#, loc='center left', bbox_to_anchor=(1, 0.5))
sns.despine()
fig.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
Path(pwd).mkdir(parents=True, exist_ok=True)
fname = "15-min-dwell1.png"
fname_pdf = "15-min-dwell1.pdf"
fig.savefig(os.path.join(pwd, fname), transparent=True)
fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%% Epoch-wise analysis


#%%
#%% between motif paired t test
for epoch in range(1, 4):
    motif_usage_ = eval("Epoch{}_motif_usage".format(epoch))
    motif_usage_control_ = eval("Epoch{}_motif_usage_ctl".format(epoch))
    motif_usage_s3d = pd.read_csv(r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv')
    print("Epoch {}".format(epoch))
    for i in range(n_cluster):
        HC = np.vstack(motif_usage_[0])[:,i]
        BD = np.vstack(motif_usage_[1])[:,i]
        s = stats.ttest_ind(HC, BD, nan_policy='omit')
        print("motif  {}\n 2 sample t-stat: {:.2f}, p-val: {:.3f}".format(i,s.statistic, s.pvalue))

        corr_HAM_D_score = scipy.stats.pearsonr(HC, HAM_D_score[:n_subject_in_population])
        corr_YMRS_score = scipy.stats.pearsonr(HC, YMRS_score[:n_subject_in_population])
        print("          YMARS: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
        print("          HAM_D: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score[0][0], corr_HAM_D_score[1]))

        # only correlate with BD list
        corr_HAM_D_score_BD = scipy.stats.pearsonr(BD, HAM_D_score[n_subject_in_population:])
        corr_YMRS_score_BD = scipy.stats.pearsonr(BD, YMRS_score[n_subject_in_population:])
        print("          YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0][0], corr_YMRS_score_BD[1]))
        print("          HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD[0][0], corr_HAM_D_score_BD[1]))

        print("  Control \n")
        HC_ctl = np.vstack(motif_usage_control_[0])[:,i]
        BD_ctl = np.vstack(motif_usage_control_[1])[:,i]
        s_ctl = stats.ttest_ind(HC_ctl, BD_ctl)
        print(" 2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s_ctl.statistic, s_ctl.pvalue))
        # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
        corr_HAM_D_score_ctl = scipy.stats.pearsonr(HC_ctl, HAM_D_score[:n_subject_in_population])
        corr_YMRS_score_ctl = scipy.stats.pearsonr(HC_ctl, YMRS_score[:n_subject_in_population])
        print("          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_ctl[0][0], corr_YMRS_score_ctl[1]))
        print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_ctl[0][0], corr_HAM_D_score_ctl[1]))

        # only correlate with BD list
        corr_HAM_D_score_BD_ctl = scipy.stats.pearsonr(BD_ctl, HAM_D_score[n_subject_in_population:])
        corr_YMRS_score_BD_ctl = scipy.stats.pearsonr(BD_ctl, YMRS_score[n_subject_in_population:])
        print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD_ctl[0][0],
                                                                                   corr_YMRS_score_BD_ctl[1]))
        print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD_ctl[0][0],
                                                                                   corr_HAM_D_score_BD_ctl[1]))

# %% Scored motif between motif paired t test and score correlation
for epoch in range(1, 4):
    motif_usage_score_ = eval("Epoch{}_motif_usage_score".format(epoch))
    print("Epoch {}".format(epoch))
    motif_usage_cat_score = np.asarray(motif_usage_cat_score)
    for i, motif_behavior in enumerate(score_bahavior_names):
        HC_score = np.vstack(motif_usage_score_[0])[:, i]
        BD_score = np.vstack(motif_usage_score_[1])[:, i]
        print(" Score motif  {}".format(motif_behavior))
        if np.sum(HC_score) == 0 or np.sum(BD_score) == 0:
            print(" ")
            # print(np.sum(HC_score) == 0)
            # print(np.sum(BD_score) == 0)

            continue
        s_score = stats.ttest_ind(HC_score, BD_score)

        print(" 2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s_score.statistic, s_score.pvalue))
        # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
        corr_HAM_D_score_score = scipy.stats.pearsonr(HC_score, HAM_D_score[:n_subject_in_population])
        corr_YMRS_score_score = scipy.stats.pearsonr(HC_score, YMRS_score[:n_subject_in_population])
        print("          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_score[0][0],
                                                                                   corr_YMRS_score_score[1]))
        print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_score[0][0],
                                                                                   corr_HAM_D_score_score[1]))

        # only correlate with BD list
        corr_HAM_D_score_BD_score = scipy.stats.pearsonr(BD_score, HAM_D_score[n_subject_in_population:])
        corr_YMRS_score_BD_score = scipy.stats.pearsonr(BD_score, YMRS_score[n_subject_in_population:])
        print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD_score[0][0],
                                                                                   corr_YMRS_score_BD_score[1]))
        print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD_score[0][0],
                                                                                   corr_HAM_D_score_BD_score[1]))

#%% Plot Box
for epoch in range(1, 4):
    motif_usage_ = eval("Epoch{}_motif_usage".format(epoch))
    motif_usage_cat = np.asarray(motif_usage_)
    states = []
    for i in range(n_cluster):
        states.append([i]*n_subject_in_population)
    states = np.asarray(states).flatten()
    sns.set_style('white')

    HC_idx = np.zeros(n_subject_in_population * n_cluster)
    BD_idx = np.ones(n_subject_in_population * n_cluster)

    ds = pd.DataFrame(np.concatenate((
        np.concatenate((motif_usage_cat[0, :, :].T.flatten(), motif_usage_cat[1, :, :].T.flatten()), 0).reshape(-1, 1),
        np.concatenate((HC_idx, BD_idx), 0).reshape(-1, 1),
        np.concatenate((states, states), 0).reshape(-1, 1)), 1),
        columns=['motif frequency', 'is_BD', 'state'])

    fig, ax = plt.subplots(1, 1, figsize=(w, 4))
    violin = sns.boxplot(y="motif frequency", x='state',hue='is_BD',
                   data=ds, orient="v", palette=sns.color_palette("tab10"))
    handles = violin.legend_.legendHandles
    dict_name = {0.0:'HC', 1.0:'BD'}
    labels = [dict_name[float(text.get_text())] for text in ax.legend_.texts]
    #sns.swarmplot(y="motif frequency", x="state", hue='is_BD',data=ds,dodge=True,size=2)
    x = np.arange(n_cluster)
    ax.legend(handles, labels)
    ax.set_xticks(x)
    ax.set_title('Epoch {} dwell frequency over {} motifs'.format(epoch, n_cluster))
    ax.set_xlabel('Motifs(States)')
    sns.despine()
    fig.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "{}-dwell.png".format(epoch)
    fname_pdf = "{}-dwell.pdf".format(epoch)
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%% Plot three bar Box

for j in range(2):
    fig, ax = plt.subplots(1, 1, figsize=(w, 4))


    motif_usage_1 = eval("Epoch{}_motif_usage".format(1))
    motif_usage_cat1 = np.asarray(motif_usage_1[j])
    motif_usage_2 = eval("Epoch{}_motif_usage".format(2))
    motif_usage_cat2 = np.asarray(motif_usage_2[j])
    motif_usage_3 = eval("Epoch{}_motif_usage".format(3))
    motif_usage_cat3 = np.asarray(motif_usage_3[j])

    label1 = np.ones(len(motif_usage_cat1[:, :].T.flatten()))
    label2 = np.ones(len(motif_usage_cat1[:, :].T.flatten())) * 2
    label3 = np.ones(len(motif_usage_cat1[:, :].T.flatten())) * 3
    states = []
    for i in range(n_cluster):
        states.append([i]*n_subject_in_population)
    states = np.asarray(states).flatten()
    sns.set_style('white')


    ds = pd.DataFrame(np.concatenate((
        np.concatenate((motif_usage_cat1[:, :].T.flatten(), motif_usage_cat2[:, :].T.flatten(), motif_usage_cat3[:, :].T.flatten()), 0).reshape(-1, 1),
        np.concatenate((label1, label2, label3), 0).reshape(-1, 1),
        np.concatenate((states, states,states), 0).reshape(-1, 1)), 1),
        columns=['motif frequency', 'epoch', 'state'])


    violin = sns.boxplot(y="motif frequency", x='state',hue='epoch',
                   data=ds, orient="v", color=b_o_colors[j])
    handles = violin.legend_.legendHandles
    dict_name = {1.0:'Epoch 1', 2.0:'Epoch 2', 3.0:'Epoch 3'}
    labels = [dict_name[float(text.get_text())] for text in ax.legend_.texts]
    #sns.swarmplot(y="motif frequency", x="state", hue='is_BD',data=ds,dodge=True,size=2)
    x = np.arange(n_cluster)
    ax.legend(handles, labels)
    ax.set_xticks(x)
    ax.set_title('{} dwell frequency over {} motifs'.format(titles[j],  n_cluster))
    ax.set_xlabel('Motifs(States)')
    sns.despine()
    fig.show()

    pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = "{}-dwell-{}.png".format(epoch, titles[j])
    fname_pdf = "{}-dwell-{}.pdf".format(epoch, titles[j])
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
#%% Plot histogram

from scipy import stats
from scipy.stats import ks_2samp
for epoch in range(1, 4):
    motif_usage_ = eval("Epoch{}_motif_usage_cat".format(epoch))
    p0 = ks_2samp(motif_usage_[0], motif_usage_[1])
    p = stats.ttest_ind(motif_usage_[0], motif_usage_[1])
    print("ks_2samp: {}, ttest:{}".format(p0, p))
    x = np.arange(n_cluster)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.bar(x, motif_usage_[0] / np.sum(motif_usage_[0]),width=0.4)
    ax.bar(x +  0.4, motif_usage_[1] / np.sum(motif_usage_[1]), width=0.4)
    ax.set_title("Epoch {}".format(epoch))
    ax.set_xlabel("motifs")
    ax.set_ylabel("% dwell time")
    ax.set_xticks(x)
    ax.set_ylim([0, 0.25])
    ax.legend(titles, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.show()


