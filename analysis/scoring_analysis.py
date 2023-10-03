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
#%%
project_name = 'BD20-Jun5-2022'
config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
n_cluster = 10
model_name = 'VAME'
control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']

#%% Read door close time from scoring (only need to run for the first time)
scoring_pth = 'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\Scoring'
start_frame = pd.read_csv('G:\start_frame.csv')

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
