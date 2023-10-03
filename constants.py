# Created by zhanq at 5/23/2023
# File:
# Description: configuration constants for project behavior, including paths, data, parameters, etc.
# Scenario:
# Usage
#%%
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy import stats
from pathlib import Path
from vame.analysis.community_analysis import read_config, compute_transition_matrices
#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
from vame.analysis.pose_segmentation import get_motif_usage
#%%  Paths

if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
else:
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
#%% Colors

b_o_colors = ['#1f77b4', '#ff7f0e']

#%%  VAME Parameters

n_cluster = 10
model_name = 'VAME'
project_name = 'BD25-HC25-final-May17-2023'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(cfg['project_path'],"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)

#%% Data

patient_information = 'video-information.csv'

patient_information = pd.read_csv(os.path.join('data', patient_information))
subject_names = patient_information['video_name'].tolist()
condition = patient_information['condition'].tolist()
bipolar_state = patient_information[['video_name', 'bipolar_state']]
BD_videos = bipolar_state[bipolar_state['bipolar_state'] == 'Euthymic']['video_name'].to_list()
control_videos = bipolar_state[bipolar_state['bipolar_state'] == 'healthy']['video_name'].to_list()
n_subject_in_population = len(control_videos)

start_frame = patient_information[['video_name','door_close']]

YMRS = patient_information[['video_name', 'YMRS']] #diagnosis_score[['Subject ID', 'YMRS (max score, 60. Pts are ineligible > 12)']]
YMRS = YMRS.set_index('video_name').T.to_dict('list')
HAM_D = patient_information[['video_name','HAMD']] #diagnosis_score[['Subject ID','HAM-D']]
HAM_D = HAM_D.set_index('video_name').T.to_dict('list')
gender_list = patient_information[['video_name', 'gender']]
gender = gender_list.set_index('video_name').T.to_dict('list')


