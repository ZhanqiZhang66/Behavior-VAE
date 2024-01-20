# Created by zhanq at 1/19/2024
# File:
# Description: decode position from the latent for sanity check
# Scenario:
# Usage

import numpy as np
import os
from vame.util.auxiliary import read_config
from data.load_data import load_pt_data
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
#%%

if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
else:
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
#%%
project_name = 'BD25-HC25-final-May17-2023'
config = r'{}\Behavior_VAE_data\{}\config.yaml'.format(onedrive_path, project_name) # config = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}/config.yaml'.format(project_name)
cfg = read_config(config)
dlc_path = os.path.join(cfg['project_path'],"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
model_name = 'VAME'
path_to_file = cfg['project_path']

data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data()
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
body_i = 39 # center of feet
# %% train decoder on


Latent_all = []
Position_all = []
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        #print(v)
        data = np.load(os.path.join(path_to_file, 'data', 'pose_sequence', v + '-90pct_seq.npy'))

        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy'))  # L x 30
        Latent_all.append(latent_vector[:15 * 60 * 30])
        # for easier visualization, we only plot the first 15 min
        data = data[temp_win // 2:-temp_win // 2]
        data = data[:15 * 60 * 30]
        Position_all.append(data[:, body_i].reshape(-1, 1))
#%%

X = np.vstack(Latent_all)
Y = np.vstack(Position_all)
rand = random.randrange(100)
train_scores = []
test_scores = []
for i in range(rand, rand + 50):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.24, random_state=i)

    mask_train = ~(np.isnan(xtrain).any(axis=1) | np.isnan(ytrain))

    mask_test = ~(np.isnan(xtest).any(axis=1) | np.isnan(ytest))

    reg = LinearRegression().fit(xtrain[mask_train], ytrain[mask_train])

    train_score = reg.score(xtrain[mask_train], ytrain[mask_train])
    test_score = reg.score(xtest[mask_test], ytest[mask_test])

    train_scores.append(train_score)
    test_scores.append(test_score)

