
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

from scipy.ndimage import label
#%%
import os, sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\deeplab-pytorch")
#%% Import DLC annotations
import pandas as pd

data0 = pd.read_csv(r'G:\hBPM_Videos\correction_videos\dlc\10584 COMT_defishDLC_resnet_50_SkeletonNov9shuffle1_200000.csv')

data = data0.to_numpy()
#%% 

input_dir = r'G:\OF_hBPM\mask'
mask_size = []
n = []
ii = 0
#for filename in glob.glob(os.path.join(input_dir, '*.*')):
#%%
datai = data[3672,:]
filename = r'G:\OF_hBPM\mask\10584_COMT_defish3670_mask.npy'
flowname = r'G:\OF_hBPM\flow\10584_COMT_defish3670_flow.npy'
framename = r'G:\OF_hBPM\frame\10584_COMT_defish3670.npy'
frame = np.load(framename)
flow = np.load(flowname)
frame = frame[:,:,::-1] # BGR to RGB
from PIL import Image
im = Image.fromarray(frame)
im.save(r"D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\framea.png")

mask = np.load(filename)
print(np.shape(mask))   # 1080 x 1920
mask[:, 0:240] = 0
mask[:, 1680:] = 0
labeled_array, num_features = label(mask)

feature_size = [np.sum(labeled_array == i) for i in range(num_features)]
feature_order = np.argsort(-np.asarray(feature_size))
# fig, ax = plt.subplots(nrows=5, ncols=1)
count = 0
# for i in feature_order[1]:
denoised = labeled_array == 1
n.append(np.sum(denoised))

B = np.argwhere(denoised)
(ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
y_mid = int((ystart + ystop)/2)
x_mid = int((xstart + xstop)/2)
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

# 	ax[count].imshow(Atrim)
# 	ax[count].set_axis_off()
# 	count += 1
#%%
import cv2
videopath = r'G:\hBPM_Videos\correction_videos\dlc\10584 COMT_defishDLC_resnet_50_SkeletonNov9shuffle1_200000_labeled.mp4'
cap = cv2.VideoCapture(videopath)
cap.set(cv2.CAP_PROP_POS_FRAMES, 367)
res, frame_dlc = cap.read()

#%%
import matplotlib.patches as patches
fig, ax = plt.subplots()
ax.imshow(im)
ax.imshow(Atrim.astype(np.float32), alpha=0.8)
rect = patches.Rectangle((xstop-350, ystop-420 ),  xstop-xstart, ystop-ystart,linewidth=5, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)
ax.axis("off")
plt.show()
#%%
fig, ax = plt.subplots()

im = plt.imread(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\frame_blur.png')
implot = ax.imshow(im)
#
x = [float(x) for x in [ datai[4], datai[7], 800,datai[13], datai[16]]]
y = [float(x) for x in [ datai[5], datai[8], 384,datai[14], datai[17]]]
ax.scatter(x, y, marker='o',s=50,c='b')
ax.axis("off")
plt.show()
#%%
from defisheye import Defisheye

dtype = 'linear'
format = 'fullframe'
fov = 180
pfov = 120

img = "./images/example3.jpg"
img_out = f"{dtype}_{format}_{pfov}_{fov}.jpg"

obj = Defisheye(im, dtype=dtype, format=format, fov=fov, pfov=pfov)
img_corr = obj.convert(img_out)
plt.imsave(img_corr, r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\frame_blur_corr.png')
#%%
import    matplotlib
# flow = flow.astype('float')
# flow[flow==0] = 'nan'
# masked_array = np.ma.array(flow, mask=np.isnan(flow))
# cmap = matplotlib.cm.jet
# cmap.set_bad('white',1.)
# ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
# plt.tight_layout()
# plt.suptitle(filename)
fig, ax = plt.subplots()
ax.imshow(flow)
ax.axis("off")
plt.show()

