# Created by zhanq at 11/6/2023
# File:
# Description:
# Scenario:
# Usage

# https://stackoverflow.com/questions/26602981/correct-barrel-distortion-in-opencv-manually-without-chessboard-image
import numpy as np
import cv2
#%%
img_path = r'C:\Users\zhanq\OneDrive - UC San Diego\Behavior_VAE_data\env.png'
src = cv2.imread(img_path)
width = src.shape[1]
height = src.shape[0]

distCoeff = np.zeros((4, 1), np.float64)

# TODO: add your coefficients here!
k1 = -2e-5;  # negative to remove barrel distortion
k2 = 0.0;
p1 = 0.0;
p2 = 0.0;

distCoeff[0, 0] = k1;
distCoeff[1, 0] = k2;
distCoeff[2, 0] = p1;
distCoeff[3, 0] = p2;

# assume unit matrix for camera
cam = np.eye(3, dtype=np.float32)

cam[0, 2] = width / 2.0  # define center x
cam[1, 2] = height / 2.0  # define center y
cam[0, 0] = 10.  # define focal length x
cam[1, 1] = 10.  # define focal length y

# here the undistortion will be computed
dst = cv2.undistort(src, cam, distCoeff)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()