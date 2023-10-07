# Created by Victoria Zhang at 8/14/2022
# File: scoring_analysis.py
# Description: Read door_close time and behavior scoring information from the xls files
# Scenario:
# Usage:
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
from data.load_data import load_pt_data
#%% Paths
if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
else:
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'

#%%
project_name = 'BD25-HC25-final-May17-2023'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name)
cfg = read_config(config)
n_cluster = 10
model_name = 'VAME'

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data()
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']

#%% Read door close time from scoring (only need to run for the first time)
scoring_pth = r'C:\Users\zhanq\OneDrive - UC San Diego\Behavior_VAE_data\Scoring\BD25-HC25-final'

for j, videos in enumerate([control_videos, BD_videos]):
    for i in range(len(videos)):
        v = videos[i]
        hBPM_score = '{}_15MIN&5MIN_hBPM.xls'.format(v)
        vs_score = '{}_15MIN&5MIN.v3.xls'.format(v)
        if os.path.exists(os.path.join(scoring_pth, hBPM_score)):
            score = os.path.join(scoring_pth, hBPM_score)
            score_ds = pd.read_excel(io=score, sheet_name='15 template data', header=None)
            door_close = score_ds.iloc[0][7][20:].split(":")
            door_close_time = (int(door_close[0]) *60 + int(door_close[1])) * 30
        elif os.path.exists(os.path.join(scoring_pth, vs_score)):
            score = os.path.join(scoring_pth, vs_score)
            score_ds = pd.read_excel(io=score, sheet_name='15 template data', header=None)
            door_close = score_ds.iloc[0][9]
            door_close_time = (door_close.minute + door_close.hour * 60) * 30
        else:
            print('Cannot find scoreing for {}'.format(v))


        v_index = start_frame.loc[start_frame['video_name'] == v].index.values[0]
        start_frame.at[v_index, 'door_close'] = int(door_close_time)
# start_frame.to_csv('G:\start_frame.csv')

#%% Read scoring of video
from openpyxl import load_workbook

scoring_path = r'C:\Users\zhanq\OneDrive - UC San Diego\Behavior_VAE_data\Scoring\BD25-HC25-final'
bahavior_names =["sit_obj", "sit", "stand_obj", "stand", "walk_obj", "walk", "lie_obj", "lie", "interact", "wear", "exercise"]
total_usage = np.zeros([2,6])
motif_usage_cat = [[],[]]

for nf, filename in enumerate(os.listdir(scoring_path)):
    if filename[0] == 'B':
        video_name = filename[:7]
        data = pd.read_excel(os.path.join(scoring_path, filename), sheet_name='15 template data', skiprows=3,
                             usecols=range(1, 12))
        usage = pd.read_excel(os.path.join(scoring_path, filename), sheet_name='15 Data time', skiprows=3, header=None,
                              usecols=range(12, 14))
    else:
        video_name = filename[:5]
        data = pd.read_excel(os.path.join(scoring_path, filename), sheet_name='template data', skiprows=3,
                             usecols=range(1, 12))
        usage = pd.read_excel(os.path.join(scoring_path, filename), sheet_name='template time', skiprows=3, header=None,
                              usecols=range(12, 14))


    print("Reading {} scoring overall motif usage time data".format(video_name))

    # Get motif usage time and category from time to # frame
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

    # Get motif usage at each frame and change format
    print("        {} motif usage time breakup".format(video_name))
    data = data.dropna(how='all')
    data_mat = pd.DataFrame.to_numpy(data)
    frame_score = []
    for i in range(int(len(data_mat) / 2) -1):
        start = data_mat[i*2, :]
        end = data_mat[i*2+1, :]
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
