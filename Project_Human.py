# Created by Victoria Zhang at 4/24/2022
# File: Project_Human.py
# Description: main function of human project
# Scenario:
# Usage:

#%%
import bound_n_rotate, analyze_dlc, dwell_time, itop, latent_visualization, scoring_analysis, transition_matrix

#%%  Flags

#%%  PREPROCESS DLC markers, skeletons
bound_n_rotate # Import DLC annotations, crop, save info, and align frames egocentrically
analyze_dlc
#%%  PREPROCESS Scorings
scoring_analysis
#%%  RUN VAME
itop
#%%  dwell time (state-wise) ANALYSIS
dwell_time
#%% transition (between states) ANALYSIS
transition_matrix
#%%  latent representation (dynamics) ANALYSIS
latent_visualization

