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
project_name = 'BD20-Feb25-2023'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(cfg['project_path'],"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
model_name = 'VAME'

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data()
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']

n_subject_in_population = len(control_videos)
start_frame = pd.read_csv(os.path.join(onedrive_path,'Behavior_VAE_data', 'start_frame_vic_50.csv'),  usecols=[0,1])
diagnosis_score = pd.read_csv(os.path.join(onedrive_path,'Behavior_VAE_data', 'start_frame_vic_50.csv'),  usecols=[0,4,5])#pd.read_csv('D:\OneDrive - UC San Diego\Behavior_VAE_data\Participant_videos_attributes\First-24-Videos\Subject_24ID-BDs-HCs-Victoria-PC.csv',encoding='windows-1252')
gender_list = pd.read_csv(os.path.join(onedrive_path,'Behavior_VAE_data', 'start_frame_vic_50.csv'),  usecols=[0,7])

YMRS = diagnosis_score[['video_name', 'YMRS']] #diagnosis_score[['Subject ID', 'YMRS (max score, 60. Pts are ineligible > 12)']]
YMRS = YMRS.set_index('video_name').T.to_dict('list') #YMRS.set_index('Subject ID').T.to_dict('list')
HAM_D = diagnosis_score[['video_name','HAMD']] #diagnosis_score[['Subject ID','HAM-D']]
HAM_D = HAM_D.set_index('video_name').T.to_dict('list') #HAM_D.set_index('Subject ID').T.to_dict('list')
gender = gender_list[['video_name','Gender']]
gender = gender.set_index('video_name').T.to_dict('list')
#%%
YMRS_score = []
HAM_D_score = []
titles = ["CP", "BD"]
N = [0, 0]
Motif_usages = [[], []]
Motif_usage_pct = [[], []]
motif_usage_cat = [[], []]
Latent_vectors = [[], []]
Labels = [[], []]
TM = [[], []]
Cluster_center = []
transition_matrices = []

Epoch1_labels = [[], []]
Epoch1_motif_usage = [[], []]
Epoch1_latent_vector = [[], []]
Epoch1_transition_matrix = [[], []]
Epoch1_motif_usage_cat = [[],[]]

Epoch2_labels = [[], []]
Epoch2_motif_usage = [[], []]
Epoch2_latent_vector = [[], []]
Epoch2_transition_matrix = [[], []]
Epoch2_motif_usage_cat = [[],[]]

Epoch3_labels = [[], []]
Epoch3_motif_usage = [[], []]
Epoch3_latent_vector = [[], []]
Epoch3_transition_matrix = [[], []]
Epoch3_motif_usage_cat = [[],[]]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        YMRS_score.append(YMRS[v][0])
        HAM_D_score.append(HAM_D[v][0])
        print("Loading {}-{} data {}/{}...".format(v, titles[j], i, len(videos)))
        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        transition_m = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        motif_usage = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 30

        v_index = start_frame.loc[start_frame['video_name'] == v].index.values[0]
        door_close_time = start_frame.loc[v_index, 'door_close']
        start_time = start_frame.loc[v_index, 'door_close']#start_frame.loc[v_index, 'n']
        five_min_frame_no = int(5 * 60 * 30)
        offset = 0 #int(start_time - door_close_time)

        epoch_1_label = label[:five_min_frame_no + offset]
        epoch_2_label = label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label = label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

        epoch_1_motif_usage = get_motif_usage(epoch_1_label, n_cluster)
        epoch_2_motif_usage = get_motif_usage(epoch_2_label, n_cluster)
        epoch_3_motif_usage = get_motif_usage(epoch_3_label, n_cluster)

        epoch_1_latent_vector = latent_vector[:five_min_frame_no + offset]
        epoch_2_latent_vector = latent_vector[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_latent_vector = latent_vector[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]

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
            tm = transition_m
            Cluster_center.append(cluster_center)
            m_e1 = epoch_1_motif_usage
            m_e2 = epoch_2_motif_usage
            m_e3 = epoch_3_motif_usage
            print(np.shape(m_e3))
        else:

            latent = np.concatenate([latent, latent_vector])
            l = np.concatenate([l,label])
            m += motif_usage
            tm += transition_m
            Cluster_center.append(cluster_center)
            m_e1 += np.asarray(epoch_1_motif_usage)
            m_e2 += np.asarray(epoch_2_motif_usage)
            m_e3 += np.asarray(epoch_3_motif_usage)
            print(np.shape(m_e3))

        motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
        num_points = latent_vector.shape[0]
        n += num_points
    Epoch1_motif_usage_cat[j] = m_e1
    Epoch2_motif_usage_cat[j] = m_e2
    Epoch3_motif_usage_cat[j] = m_e3
    N[j] = n
    Motif_usages[j] = m
    Motif_usage_pct[j] = m/n
    Latent_vectors[j] = latent
    Labels[j] = l
    TM[j] = tm/len(videos)

#%% Population-wise analysis
#%% between motif paired t test and score correltion
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
motif_usage_cat = np.asarray(motif_usage_cat)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for i in range(n_cluster):
    CP = motif_usage_cat[0,:,i].reshape(-1,1)
    BD = motif_usage_cat[1,:,i].reshape(-1,1)
    s = stats.ttest_ind(CP, BD)
    # because our statistic is vectorized, we pass `vectorized=True`
    # `n_resamples=np.inf` indicates that an exact test is to be performed
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
    res = permutation_test((CP, BD), statistic, permutation_type='sampled', vectorized=True,
                           n_resamples=np.inf, alternative='less')
    print("motif  {}, 2 sample t-stat: {:.2f}, p-val: {:.3f}".format(i,s.statistic[0], s.pvalue[0]))
    print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score = scipy.stats.pearsonr(motif_usage_cat[0,:,i], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score= scipy.stats.pearsonr(motif_usage_cat[0,:,i], YMRS_score[:n_subject_in_population])
    print("          YMARS-CP: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0], corr_YMRS_score[1]))
    print("          HAM_D-CP: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score[0], corr_HAM_D_score[1]))


    # only correlate with BD list
    corr_HAM_D_score_BD = scipy.stats.pearsonr(motif_usage_cat[1,:,i], HAM_D_score[n_subject_in_population:])
    corr_YMRS_score_BD = scipy.stats.pearsonr(motif_usage_cat[1,:,i], YMRS_score[n_subject_in_population:])
    print("          YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0], corr_YMRS_score_BD[1]))
    print("          HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score_BD[0], corr_HAM_D_score_BD[1]))
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
    print("Epoch {}".format(epoch))
    for i in range(n_cluster):
        CP = np.vstack(motif_usage_[0])[:,i]
        BD = np.vstack(motif_usage_[1])[:,i]
        s = stats.ttest_ind(CP, BD, nan_policy='omit')
        print("  motif  {}, t-stat: {:.2f}, p-val: {:.3f}".format(i,s.statistic, s.pvalue))

        corr_HAM_D_score = scipy.stats.pearsonr(CP, HAM_D_score[:n_subject_in_population])
        corr_YMRS_score = scipy.stats.pearsonr(CP, YMRS_score[:n_subject_in_population])
        print("          YMARS: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0], corr_YMRS_score[1]))
        print("          HAM_D: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score[0], corr_HAM_D_score[1]))

        # only correlate with BD list
        corr_HAM_D_score_BD = scipy.stats.pearsonr(BD, HAM_D_score[n_subject_in_population:])
        corr_YMRS_score_BD = scipy.stats.pearsonr(BD, YMRS_score[n_subject_in_population:])
        print("          YMARS-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score_BD[0], corr_YMRS_score_BD[1]))
        print("          HAM_D-BD: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score_BD[0], corr_HAM_D_score_BD[1]))
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


