# Created by Victoria Zhang at 4/3/2022
# File: read_skeleton_json.py
# Description: read open pose labeled skeleton coordinates and plot the sequence
# Scenario:
# Usage:
#%%
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print('OpenCV - version: ', cv2.__version__)
import pandas as pd
#%%
# video file
cap = cv2.VideoCapture(r'G:\hBPM_Videos\correction_videos\10584_COMT_defish_trimed.mpg')


def get_vid_properties():
    width = int(cap.get(3))  # float
    height = int(cap.get(4))  # float
    cap.release()
    return width, height


print('Video Dimensions: ', get_vid_properties())
#%%
# Load keypoint data from JSON output
column_names = ['x', 'y', 'acc']

# Paths - should be the folder where Open Pose JSON output was stored
path_to_json = r"G:\openpose\output_jsons"

# Import Json files, pos_json = position JSON
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print('Found: ', len(json_files), 'json keypoint frame files')
count = 0

width, height = get_vid_properties()

# instanciate dataframes
body_keypoints_df = pd.DataFrame()
left_knee_df = pd.DataFrame()
json0 = json.load(open(path_to_json + '\\' + json_files[0]))

print('example json files: ', json_files[0])
#%%
# Loop through all json files in output directory
# Each file is a frame in the video
# If multiple people are detected - choose the most centered high confidence points
body_parts_name = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
temp = {key: [] for key in body_parts_name}
for file in json_files:

    temp_df = json.load(open(path_to_json + '\\' + file))

    # no person is detected
    if len(temp_df['people']) == 0:
        for k in temp:
            temp[k].append((0,0,0))
    else:
        data = temp_df['people'][0]['pose_keypoints_2d']
        for idx, (i, j, k) in enumerate(zip(data[0::3], data[1::3], data[2::3])):
            body_part = body_parts_name[idx]

            temp[body_part].append((i, j, k))
    tmp = temp[body_part]
    temp[body_part] = pd.DataFrame(tmp)

#%% plot temoral structure of each body part

time = np.linspace(1,len(temp['Nose']),num=len(temp['Nose']))
for k in temp:
    body_part = k
    x = np.asarray(temp[k].iloc[:, 0])
    y = np.asarray(temp[k].iloc[:, 1])
    c = np.asarray(temp[k].iloc[:, 2])
    good_labels = np.where(c >= 0.7)
    bad_labels = np.where(c <= 0.7)
    fig, axs = plt.subplots(3, 1)
    axs[0].set_title(body_part)
    axs[0].plot(time, x, time, y)
    axs[0].set_xlabel('frames')
    axs[0].set_ylabel('coordinates')

    axs[1].set_title("c>0.7, N={}/{}".format(len(good_labels[0]), len(time)))
    axs[1].scatter(time[good_labels], x[good_labels], s=2, label='x')
    axs[1].scatter(time[good_labels], y[good_labels], s=2, label='y')
    axs[1].legend(loc="upper right")
    axs[1].set_xlabel('frames')
    axs[1].set_ylabel('coordinates')
    axs[1].grid(True)

    img = mpimg.imread(r'G:\\env.png')
    axs[2].imshow(img)
    axs[2].plot(x[good_labels], y[good_labels])
    plt.show()
    pwd = r'G:\\openpose\\temporal_dynamic_figures\\'
    fig.savefig('{}{}_{}.png'.format(pwd, '10584_COMT', body_part))

#%%

