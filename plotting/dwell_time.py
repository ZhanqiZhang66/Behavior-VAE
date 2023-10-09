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
from scipy.stats import permutation_test
from pathlib import Path
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
from data.load_data import load_pt_data
#%%
if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
else:
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
#%%
b_o_colors = ['#1f77b4', '#ff7f0e']
#%%
project_name = 'BD25-HC25-final-May17-2023'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(cfg['project_path'],"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
model_name = 'VAME'

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data()
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']

n_subject_in_population = len(control_videos)
# start_frame = pd.read_csv(os.path.join(onedrive_path,'Behavior_VAE_data', 'start_frame_vic_50.csv'),  usecols=[0,1])
# diagnosis_score = pd.read_csv(os.path.join(onedrive_path,'Behavior_VAE_data', 'start_frame_vic_50.csv'),  usecols=[0,4,5])#pd.read_csv('D:\OneDrive - UC San Diego\Behavior_VAE_data\Participant_videos_attributes\First-24-Videos\Subject_24ID-BDs-HCs-Victoria-PC.csv',encoding='windows-1252')
# gender_list = pd.read_csv(os.path.join(onedrive_path,'Behavior_VAE_data', 'start_frame_vic_50.csv'),  usecols=[0,7])
#
# YMRS = diagnosis_score[['video_name', 'YMRS']] #diagnosis_score[['Subject ID', 'YMRS (max score, 60. Pts are ineligible > 12)']]
# YMRS = YMRS.set_index('video_name').T.to_dict('list') #YMRS.set_index('Subject ID').T.to_dict('list')
# HAM_D = diagnosis_score[['video_name','HAMD']] #diagnosis_score[['Subject ID','HAM-D']]
# HAM_D = HAM_D.set_index('video_name').T.to_dict('list') #HAM_D.set_index('Subject ID').T.to_dict('list')
# gender = gender_list[['video_name','Gender']]
# gender = gender.set_index('video_name').T.to_dict('list')
#%%
YMRS_score = []
HAM_D_score = []
titles = ["CP", "BD"]

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
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        YMRS_score.append(YMRS[v])
        HAM_D_score.append(HAM_D[v])
        print("Loading {}-{} data {}/{}...".format(v, titles[j], i+1, len(videos)))
        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        motif_usage = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))

        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")

        control_label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        control_cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\DLC_{}_cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, n_cluster, v))
        control_motif_usage = get_motif_usage(control_label, n_cluster)

        door_close_time = start_frame[v]
        start_time = start_frame[v] #start_frame.loc[v_index, 'n']
        five_min_frame_no = int(5 * 60 * 30)
        offset = 0 #int(start_time - door_close_time)

        # concat labels, motif usage for 3 epochs
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

        # concat labels, motif usage for 3 epochs for control
        epoch_1_label_ctl = control_label[:five_min_frame_no + offset]
        epoch_2_label_ctl = control_label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label_ctl = control_label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

        epoch_1_motif_usage_ctl = get_motif_usage(epoch_1_label_ctl, n_cluster)
        epoch_2_motif_usage_ctl = get_motif_usage(epoch_2_label_ctl, n_cluster)
        epoch_3_motif_usage_ctl = get_motif_usage(epoch_3_label_ctl, n_cluster)

        Epoch1_labels_ctl[j].append(epoch_1_label_ctl)
        Epoch1_motif_usage_ctl[j].append(epoch_1_motif_usage_ctl/ np.sum(epoch_1_motif_usage_ctl))

        Epoch2_labels_ctl[j].append(epoch_2_label_ctl)
        Epoch2_motif_usage_ctl[j].append(epoch_2_motif_usage_ctl/ np.sum(epoch_2_motif_usage_ctl))

        Epoch3_labels_ctl[j].append(epoch_3_label)
        Epoch3_motif_usage_ctl[j].append(epoch_3_motif_usage_ctl/ np.sum(epoch_3_motif_usage_ctl))


        if i == 0:
            l = label
            l_control = control_label
            m = motif_usage
            m_control = control_motif_usage
            Cluster_center.append(cluster_center)
            m_e1 = epoch_1_motif_usage
            m_e2 = epoch_2_motif_usage
            m_e3 = epoch_3_motif_usage

            Cluster_center_ctl.append(control_cluster_center)
            m_e1_ctl = epoch_1_motif_usage_ctl
            m_e2_ctl = epoch_2_motif_usage_ctl
            m_e3_ctl = epoch_3_motif_usage_ctl
            # print(np.shape(m_e3))
        else:
            l = np.concatenate([l,label])
            m += motif_usage
            Cluster_center.append(cluster_center)
            m_e1 += np.asarray(epoch_1_motif_usage)
            m_e2 += np.asarray(epoch_2_motif_usage)
            m_e3 += np.asarray(epoch_3_motif_usage)

            l_control = np.concatenate([l_control, control_label])
            m_control += control_motif_usage
            Cluster_center_ctl.append(control_cluster_center)
            m_e1_ctl += np.asarray(epoch_1_motif_usage_ctl)
            m_e1_ctl += np.asarray(epoch_2_motif_usage_ctl)
            m_e1_ctl += np.asarray(epoch_3_motif_usage_ctl)
            # print(np.shape(m_e3))

        motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
        motif_usage_cat_ctl[j].append(control_motif_usage / np.sum(control_motif_usage))

        num_points = label.shape[0]
        n += num_points

    Epoch1_motif_usage_cat[j] = m_e1
    Epoch2_motif_usage_cat[j] = m_e2
    Epoch3_motif_usage_cat[j] = m_e3
    Motif_usages[j] = m
    Motif_usage_pct[j] = m/n
    Labels[j] = l

    Epoch1_motif_usage_cat_ctl[j] = m_e1_ctl
    Epoch2_motif_usage_cat_ctl[j] = m_e2_ctl
    Epoch3_motif_usage_cat_ctl[j] = m_e3_ctl
    Motif_usages_ctl[j] = m_control
    Motif_usage_pct_ctl[j] = m_control / n
    Labels_ctl[j] = l_control
#%% Population-wise analysis
#%% between motif paired t test and score correltion
import permutation_test as p
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
motif_usage_cat = np.asarray(motif_usage_cat)
motif_usage_cat_ctl = np.asarray(motif_usage_cat_ctl)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i in range(n_cluster):
    CP = motif_usage_cat[0,:,i].reshape(-1,1)
    BD = motif_usage_cat[1,:,i].reshape(-1,1)
    s = stats.ttest_ind(CP, BD)
    # because our statistic is vectorized, we pass `vectorized=True`
    # `n_resamples=np.inf` indicates that an exact test is to be performed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
    # res = p.permutation_test(BD, CP)
    print("motif  {}\n2 sample t-stat: {:.2f}, p-val: {:.3f}".format(i,s.statistic[0], s.pvalue[0]))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score = scipy.stats.pearsonr(motif_usage_cat[0,:,i], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score = scipy.stats.pearsonr(motif_usage_cat[0,:,i], YMRS_score[:n_subject_in_population])
    print("          Pearson corr YMARS-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
    print("          Pearson corr HAM_D-CP: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score[0][0], corr_HAM_D_score[1]))

    # only correlate with BD list
    corr_HAM_D_score_BD = scipy.stats.pearsonr(motif_usage_cat[1,:,i], HAM_D_score[n_subject_in_population:])
    corr_YMRS_score_BD = scipy.stats.pearsonr(motif_usage_cat[1,:,i], YMRS_score[n_subject_in_population:])
    print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0][0], corr_YMRS_score_BD[1]))
    print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_BD[0][0], corr_HAM_D_score_BD[1]))

    CP_ctl = motif_usage_cat_ctl[0,:,i].reshape(-1,1)
    BD_ctl = motif_usage_cat_ctl[1,:,i].reshape(-1,1)
    s_ctl = stats.ttest_ind(CP_ctl, BD_ctl)
    # because our statistic is vectorized, we pass `vectorized=True`
    # `n_resamples=np.inf` indicates that an exact test is to be performed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
    # res = p.permutation_test(BD, CP)
    print("Control \n")
    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s_ctl.statistic[0], s_ctl.pvalue[0]))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[0,:,i], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[0,:,i], YMRS_score[:n_subject_in_population])
    print("          Pearson corr YMARS-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_ctl[0][0], corr_YMRS_score_ctl[1]))
    print("          Pearson corr HAM_D-CP: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_ctl[0][0], corr_HAM_D_score_ctl[1]))

    # only correlate with BD list
    corr_HAM_D_score_BD_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[1,:,i], HAM_D_score[n_subject_in_population:])
    corr_YMRS_score_BD_ctl = scipy.stats.pearsonr(motif_usage_cat_ctl[1,:,i], YMRS_score[n_subject_in_population:])
    print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD_ctl[0][0], corr_YMRS_score_BD_ctl[1]))
    print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_BD_ctl[0][0], corr_HAM_D_score_BD_ctl[1]))

#%% Plot Box
states = []
for i in range(n_cluster):
    states.append([i]*n_subject_in_population)
states = np.asarray(states).flatten()
sns.set_style("white")

CP_idx = np.zeros(n_subject_in_population * n_cluster)
BD_idx = np.ones(n_subject_in_population * n_cluster)

ds = pd.DataFrame(np.concatenate((
    np.concatenate((motif_usage_cat[0,:,:].T.flatten(), motif_usage_cat[1,:,:].T.flatten()), 0).reshape(-1, 1),
    np.concatenate((CP_idx, BD_idx), 0).reshape(-1, 1),
    np.concatenate((states, states), 0).reshape(-1, 1)), 1),
    columns=['motif frequency','is_BD','state'])
w = n_cluster/10 * 6
fig, ax = plt.subplots(1, 1, figsize=(w, 4))
violin = sns.boxplot(y="motif frequency", x='state',hue='is_BD',
               data=ds, orient="v", palette=sns.color_palette("tab10"))
handles = violin.legend_.legendHandles
dict_name = {0.0:'CP', 1.0:'BD'}
labels = [dict_name[float(text.get_text())] for text in ax.legend_.texts]
# sns.swarmplot(y="motif frequency", x="state", hue='is_BD',data=ds,dodge=True,size=2)
x = np.arange(n_cluster)
ax.legend(handles, labels)
ax.set_xticks(x)
ax.set_title('15 min dwell frequency over {} motifs'.format(n_cluster))
ax.set_xlabel('Motifs(States)')
sns.despine()
fig.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
Path(pwd).mkdir(parents=True, exist_ok=True)
fname = "15-min-dwell.png"
fname_pdf = "15-min-dwell.pdf"
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
    print("Epoch {}".format(epoch))
    for i in range(n_cluster):
        CP = np.vstack(motif_usage_[0])[:,i]
        BD = np.vstack(motif_usage_[1])[:,i]
        s = stats.ttest_ind(CP, BD, nan_policy='omit')
        print("motif  {}\n 2 sample t-stat: {:.2f}, p-val: {:.3f}".format(i,s.statistic, s.pvalue))

        corr_HAM_D_score = scipy.stats.pearsonr(CP, HAM_D_score[:n_subject_in_population])
        corr_YMRS_score = scipy.stats.pearsonr(CP, YMRS_score[:n_subject_in_population])
        print("          YMARS: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
        print("          HAM_D: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score[0][0], corr_HAM_D_score[1]))

        # only correlate with BD list
        corr_HAM_D_score_BD = scipy.stats.pearsonr(BD, HAM_D_score[n_subject_in_population:])
        corr_YMRS_score_BD = scipy.stats.pearsonr(BD, YMRS_score[n_subject_in_population:])
        print("          YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0][0], corr_YMRS_score_BD[1]))
        print("          HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD[0][0], corr_HAM_D_score_BD[1]))

        print("  Control \n")
        CP_ctl = np.vstack(motif_usage_control_[0])[:,i]
        BD_ctl = np.vstack(motif_usage_control_[1])[:,i]
        s_ctl = stats.ttest_ind(CP_ctl, BD_ctl)
        print(" 2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s_ctl.statistic, s_ctl.pvalue))
        # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
        corr_HAM_D_score_ctl = scipy.stats.pearsonr(CP_ctl, HAM_D_score[:n_subject_in_population])
        corr_YMRS_score_ctl = scipy.stats.pearsonr(CP_ctl, YMRS_score[:n_subject_in_population])
        print("          Pearson corr YMARS-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_ctl[0][0], corr_YMRS_score_ctl[1]))
        print("          Pearson corr HAM_D-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_ctl[0][0], corr_HAM_D_score_ctl[1]))

        # only correlate with BD list
        corr_HAM_D_score_BD_ctl = scipy.stats.pearsonr(BD_ctl, HAM_D_score[n_subject_in_population:])
        corr_YMRS_score_BD_ctl = scipy.stats.pearsonr(BD_ctl, YMRS_score[n_subject_in_population:])
        print("          Pearson corr YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD_ctl[0][0],
                                                                                   corr_YMRS_score_BD_ctl[1]))
        print("          Pearson corr HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD_ctl[0][0],
                                                                                   corr_HAM_D_score_BD_ctl[1]))
#%% Plot Box
for epoch in range(1, 4):
    motif_usage_ = eval("Epoch{}_motif_usage".format(epoch))
    motif_usage_cat = np.asarray(motif_usage_)
    states = []
    for i in range(n_cluster):
        states.append([i]*n_subject_in_population)
    states = np.asarray(states).flatten()
    sns.set_style('white')

    CP_idx = np.zeros(n_subject_in_population * n_cluster)
    BD_idx = np.ones(n_subject_in_population * n_cluster)

    ds = pd.DataFrame(np.concatenate((
        np.concatenate((motif_usage_cat[0, :, :].T.flatten(), motif_usage_cat[1, :, :].T.flatten()), 0).reshape(-1, 1),
        np.concatenate((CP_idx, BD_idx), 0).reshape(-1, 1),
        np.concatenate((states, states), 0).reshape(-1, 1)), 1),
        columns=['motif frequency', 'is_BD', 'state'])

    fig, ax = plt.subplots(1, 1, figsize=(w, 4))
    violin = sns.boxplot(y="motif frequency", x='state',hue='is_BD',
                   data=ds, orient="v", palette=sns.color_palette("tab10"))
    handles = violin.legend_.legendHandles
    dict_name = {0.0:'CP', 1.0:'BD'}
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


