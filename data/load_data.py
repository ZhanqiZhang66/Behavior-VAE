# Created by zhanq at 10/4/2023
# File:
# Description: load video, video information, pt information and other data all at once
# Scenario:
# Usage
import pandas as pd
import numpy as np
#%%
def load_pt_data(video_information_pth=r'data\video-information.csv'):
    data = pd.read_csv(video_information_pth,
                              usecols=["video_name"	,"door_close", "condition",	"bipolar_state", "YMRS", "HAMD", "gender"])

    # data = data.set_index('video_name').T.to_dict('list')
    start_frame_list = pd.read_csv(r'data\video-information.csv',
                              usecols=["video_name"	,"door_close"])
    condition_list = pd.read_csv(r'data\video-information.csv',
                                  usecols=["video_name", "condition"])
    isBD_list = pd.read_csv(r'data\video-information.csv',
                                 usecols=["video_name", "bipolar_state"])
    diagnosis_scale_list = pd.read_csv(r'data\video-information.csv',
                                  usecols=["video_name", "YMRS", "HAMD"])
    gender_list = pd.read_csv(r'data\video-information.csv',
                                  usecols=["video_name", "gender"])

    YMRS = diagnosis_scale_list[
        ['video_name', 'YMRS']]  # diagnosis_score[['Subject ID', 'YMRS (max score, 60. Pts are ineligible > 12)']]
    YMRS = YMRS.set_index('video_name').T.to_dict('list')  # YMRS.set_index('Subject ID').T.to_dict('list')

    HAM_D = diagnosis_scale_list[['video_name', 'HAMD']]  # diagnosis_score[['Subject ID','HAM-D']]
    HAM_D = HAM_D.set_index('video_name').T.to_dict('list')  # HAM_D.set_index('Subject ID').T.to_dict('list')

    gender = gender_list[['video_name', 'gender']]
    gender = gender.set_index('video_name').T.to_dict('list')

    start_frame = start_frame_list[['video_name', 'door_close']]
    start_frame = start_frame.set_index('video_name').T.to_dict('list')

    condition = condition_list[['video_name', 'condition']]
    condition = condition.set_index('video_name').T.to_dict('list')

    isBD = isBD_list[['video_name', 'bipolar_state']]
    isBD = isBD.set_index('video_name').T.to_dict('list')

    return data, YMRS, HAM_D, gender, start_frame, condition, isBD
