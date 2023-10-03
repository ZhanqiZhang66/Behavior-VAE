# Created by Victoria Zhang at 10/6/2022
# File: trajectory_heatmap.py
# Description: plot trajectory heatmap of each subject or in population
# Scenario:
# Usage:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
from pylab import imread,subplot,imshow,show
import cv2
#%% Define Project
project_name = 'BD20-Jun5-2022'
n_cluster = 10
path_to_file= 'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}'.format(project_name)
start_frame = pd.read_csv('G:\start_frame.csv')
start_frame = start_frame.set_index('video_name').T.to_dict('list')
model_name = 'VAME'
control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
pop = ['CP','BD']
#%%
background = imread(r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\background.png')
background = cv2.flip(background, 0)
#%% Read DLC readings
csv_path = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}\videos\pose_estimation'.format(project_name)
confidence = 0.9

for j, videos in enumerate([control_videos, BD_videos]):
    for p in range(len(videos)):
        v = videos[p]
        data = pd.read_csv(os.path.join(path_to_file, 'videos', 'pose_estimation', '{}.csv'.format(v)), skiprows=2)
        # data_mat = pd.DataFrame
        f_start_frame = start_frame[v][0]
        data_mat = data[data['coords'] > f_start_frame]
        data_mat = data_mat.drop(columns=['coords'])
        # get the coordinates for alignment from data table

        for i in range(int(data_mat.shape[1] / 3)):
            temp = data_mat.iloc[:, i * 3:(i + 1) * 3]
            for idx, row in temp.iterrows():
                row[1] = 480-row[1]
                if row[2] <= confidence:
                    row[0], row[1] = np.nan, np.nan
            pose_list = temp.iloc[:,0:2] if i == 0 else pd.concat([pose_list, temp.iloc[:, :2]],axis=1)

        if j == 0:
            pose_cat_CP = pose_list if p == 0 else pd.concat((pose_cat_CP, pose_list), axis=0, ignore_index=True)
        if j == 1:
            pose_cat_BD = pose_list if p == 0 else pd.concat((pose_cat_BD, pose_list), axis=0, ignore_index=True)


        #Plot heatmap of each participant
        fig = sns.displot(pose_list, x="x.13", y="y.13", height=4.8, aspect=7.2/4.8,)
        plt.imshow(background, alpha=0.2)
        plt.xlim(1, 720)
        plt.ylim(1, 480)
        plt.xlabel('center of hip x')
        plt.ylabel('center of hip y')
        plt.title("{}-{}-trajectory-heatmap".format(pop[j], v))
        plt.show()
        pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\heatmap'
        fname = "{}-{}-{}-trajectory-heatmap.png".format(pop[j], v,n_cluster)
        fig.savefig(os.path.join(pwd, fname))

        for k in range(2):
            fig = sns.displot(pose_list[(len(pose_list)//2 -1) * k : (len(pose_list)//2 -1)* (k+1)], x="x.13", y="y.13", height=4.8, aspect=7.2/4.8,)
            plt.imshow(background, alpha=0.2)
            plt.xlim(1, 720)
            plt.ylim(1, 480)
            plt.xlabel('center of hip x')
            plt.ylabel('center of hip y')
            plt.title("{}-{}-{}half-trajectory-heatmap".format(pop[j], v, k+1))
            plt.show()
            pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\heatmap'
            fname = "{}-{}-{}-{}half-trajectory-heatmap.png".format(pop[j], v, n_cluster, k+1)
            fig.savefig(os.path.join(pwd, fname))

    pose_cat = eval('pose_cat_{}'.format(pop[j]))


    fig = sns.displot(pose_cat, x="x.13", y="y.13", height=4.8, aspect=7.2 / 4.8)
    plt.imshow(background, alpha=0.2)
    plt.xlim(1, 720)
    plt.ylim(1, 480)
    plt.xlabel('center of hip x')
    plt.ylabel('center of hip y')
    plt.title("{}-population-trajectory-heatmap".format(pop[j]))
    plt.show()
    pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\heatmap'
    fname = "{}-population-{}-trajectory-heatmap.png".format(pop[j], n_cluster)
    fig.savefig(os.path.join(pwd, fname))

    for k in range(2):
        fig = sns.displot(pose_cat[(len(pose_cat) // 2 - 1) * k: (len(pose_cat) // 2 - 1) * (k + 1)], x="x.13",
                          y="y.13", height=4.8, aspect=7.2 / 4.8, )
        plt.imshow(background, alpha=0.2)
        plt.xlim(1, 720)
        plt.ylim(1, 480)
        plt.xlabel('center of hip x')
        plt.ylabel('center of hip y')
        plt.title("{}-population-{}half-trajectory-heatmap".format(pop[j], k))
        plt.show()
        pwd = r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\heatmap'
        fname = "{}-population-{}-{}half-trajectory-heatmap.png".format(pop[j], n_cluster, k)
        fig.savefig(os.path.join(pwd, fname))
#%% Plot heatmap of each participant,
