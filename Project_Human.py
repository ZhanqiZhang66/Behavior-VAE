# Created by Victoria Zhang at 4/24/2022
# File: Project_Human.py
# Description: main function of human project
# Scenario:
# Usage:
#%%
from analysis import bound_n_rotate, scoring_analysis
from plotting import analyze_dlc, dwell_time, latent_visualization, transition_matrix, sliding_window
from model import run_vame
#%%  Flags

#%%  PREPROCESS DLC markers, skeletons
bound_n_rotate # Import DLC annotations, crop, save info, and align frames egocentrically
analyze_dlc
#%%  PREPROCESS Scorings
scoring_analysis
#%%  RUN VAME
run_vame
#%%  dwell time (state-wise) ANALYSIS
dwell_time
#%% transition (between states) ANALYSIS
transition_matrix
#%%  latent representation (dynamics) ANALYSIS
latent_visualization

