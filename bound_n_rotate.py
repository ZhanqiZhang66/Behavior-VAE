# Created by Victoria Zhang at 3/17/2022
# File: bound_n_rotate.py
# Description: detect human, crop frm
# Scenario:
# Usage:

import os
import numpy as np
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt
import pandas as pd


def align_human(path_to_file, filename, video_format, crop_size, pose_list,
                pose_ref_index, confidence, pose_flip_ref, bg, frame_count, use_video=True):
    # returns: list of cropped images (if video is used) and list of cropped DLC points
    #
    # parameters:
    # path_to_file: directory
    # filename: name of video file without format
    # video_format: format of video file
    # crop_size: tuple of x and y crop size
    # dlc_list: list of arrays containg corresponding x and y DLC values
    # dlc_ref_index: indices of 2 lists in dlc_list to align mouse along
    # dlc_flip_ref: indices of 2 lists in dlc_list to flip mouse if flip was false
    # bg: background image to subtract
    # frame_count: number of frames to align
    # use_video: boolean if video should be cropped or DLC points only

    images = []
    points = []

    for i in pose_list:
        for j in i:
            if j[2] <= confidence:
                j[0], j[1] = np.nan, np.nan

    for i in pose_list:
        i = interpol(i)

    if use_video:
        capture = cv.VideoCapture(os.path.join(path_to_file, 'videos', filename + video_format))

        if not capture.isOpened():
            raise Exception(
                "Unable to open video file: {0}".format(os.path.join(path_to_file, 'videos', filename + video_format)))

    for idx in tqdm.tqdm(range(frame_count), disable=not True, desc='Align frames'):

        if use_video:
            # Read frame
            try:
                ret, frame = capture.read()
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = frame - bg
                frame[frame <= 0] = 0
            except:
                print("Couldn't find a frame in capture.read(). #Frame: %d" % idx)
                continue
        else:
            frame = np.zeros((1, 1))

        # Read coordinates and add border
        pose_list_bordered = []

        for i in pose_list:
            pose_list_bordered.append((int(i[idx][0] + crop_size[0]), int(i[idx][1] + crop_size[1])))

        img = cv.copyMakeBorder(frame, crop_size[1], crop_size[1], crop_size[0], crop_size[0], cv.BORDER_CONSTANT, 0)

        punkte = []
        for i in pose_ref_index:
            coord = []
            coord.append(pose_list_bordered[i][0])
            coord.append(pose_list_bordered[i][1])
            punkte.append(coord)
        punkte = [punkte]
        punkte = np.asarray(punkte)

        # calculate minimal rectangle around snout and tail
        rect = cv.minAreaRect(punkte)

        # change size in rect tuple structure to be equal to crop_size
        lst = list(rect)
        lst[1] = crop_size
        rect = tuple(lst)

        center, size, theta = rect

        # crop image
        out, shifted_points = crop_and_flip(rect, img, pose_list_bordered, pose_flip_ref)

        if use_video:  # for memory optimization, just save images when video is used.
            images.append(out)
        points.append(shifted_points)

    if use_video:
        capture.release()

    time_series = np.zeros((len(pose_list) * 2, frame_count))
    for i in range(frame_count):
        idx = 0
        for j in range(len(pose_list)):
            time_series[idx:idx + 2, i] = points[i][j]
            idx += 2

    return images, points, time_series

def read_align_information(dlc_dir, filename, pose_ref_index, crop_size, save_dir):
    # read out data
    data = pd.read_csv(os.path.join(dlc_dir, 'videos', 'pose_estimation', filename + '.csv'), skiprows=2)
    data_mat = pd.DataFrame.to_numpy(data)
    data_mat = data_mat[:, 1:]

    # get the coordinates for alignment from data table
    pose_list = []

    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3:(i + 1) * 3])

    # list of reference coordinate indices for alignment
    # 0: l arm, 1: r arm, 2: l foot,
    # 3: r foot, 4: head, 5: center of body
    frame_count = len(data)

    pose_ref_index = pose_ref_index

    # list of 2 reference coordinate indices for avoiding flipping
    pose_flip_ref = pose_ref_index

    frames, n, time_series = align_human(dlc_dir, filename, video_format, crop_size, pose_list, pose_ref_index,
                                         confidence, pose_flip_ref, bg, frame_count, use_video)
#%% analyze average human size
n = []
dir_name = r'G:\OF_hBPM\mask'
frame_dir_name = r'G:\OF_hBPM\frame'
for base_filename in os.listdir(dir_name):
    if base_filename.endswith('.npy'):
        this_mask_name = os.path.join(dir_name, base_filename)
        this_frame_name = os.path.join(frame_dir_name, base_filename[:-9] + '.npy')
        mask = np.load(this_mask_name)
        frame = np.load(this_frame_name)
        frame = frame[:, :, ::-1]  # BGR to RGB
        #print(np.shape(mask))  # 1080 x 1920
        mask[:, 0:240] = 0
        mask[:, 1680:] = 0
        labeled_array, num_features = label(mask)

        feature_size = [np.sum(labeled_array == i) for i in range(num_features)]
        feature_order = np.argsort(-np.asarray(feature_size))
        # fig, ax = plt.subplots(nrows=5, ncols=1)
        count = 0
        # for i in feature_order[1]:
        denoised = labeled_array == feature_order[1]
        n.append(np.sum(denoised))

        B = np.argwhere(denoised)
        (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
        y_mid = int((ystart + ystop) / 2)
        x_mid = int((xstart + xstop) / 2)
        # y_start = y_mid-100
        # y_end = y_mid +100
        # if y_mid - 100 < 0:
        # 	y_start = 0
        # 	y_end = 200
        # if y_mid + 100 > 1080:
        # 	y_end = 1080
        # 	y_start = 880
        # Atrim = denoised[y_start:y_end, x_mid-100:x_mid+100]
        Atrim = denoised
        Amask = np.repeat(Atrim[:, :, np.newaxis], 3, axis=2)
        Cframe = frame ** Amask
        fig = plt.figure()
        plt.imshow(frame)
        plt.imshow(Cframe, alpha=0.8)
        plt.title(base_filename)
        plt.show()
        pwd = r'G:\OF_hBPM\boundingbox\\'
        if not os.path.exists(pwd):
            os.makedirs(pwd)
        fig.savefig('{}{}_{}.png'.format(pwd, base_filename[:-9], 'box'))




#%%
