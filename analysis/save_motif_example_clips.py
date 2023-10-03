# Created by Victoria Zhang at 10/5/2022
# File: save_motif_example_clips.py
# Description: for each motif, save up to 20 example clips to generate descriptionsD:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022
# Scenario:
# Usage: (may need to use debug mode if the
import os, sys
from os.path import join
import numpy as np
import vame
import pandas as pd
from vame.analysis.community_analysis import read_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\VAME")  # Victoria local
sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\markov-chain")
project_name = 'BD20-Jun5-2022'
config = 'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}\config.yaml'.format(project_name)
start_frame = pd.read_csv('G:\start_frame.csv')
start_frame = start_frame.set_index('video_name').T.to_dict('list')
vame.motif_videos(config, start_frame, videoType='.mpg')
