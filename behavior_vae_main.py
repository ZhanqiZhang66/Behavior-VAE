# Created by Victoria Zhang at 4/24/2022
# File: behavior_vae_main.py
# Description: Main function of human project

from analysis import bound_n_rotate, scoring_analysis
from plotting import analyze_dlc, dwell_time, latent_visualization, transition_matrix, sliding_window
from analysis.Classifiers import classifier2
from analysis.Classifiers.Generation import motif_usage
from analysis import compute_Wasserstein_distances, plot_Wasserstein_distance
from model import run_vame


def main():
    # Define global flags for each section
    run_dlc_preprocessing = True
    run_scoring_analysis = True
    run_vame_model = True
    run_dwell_time_analysis = True
    run_transition_matrix_analysis = True
    run_latent_visualization = True
    run_continuous_metric = True
    run_classificaction = True
    run_cv_motif_usage = True

    # PREPROCESS DLC markers, skeletons
    if run_dlc_preprocessing:
        print("Running DLC marker preprocessing...")
        bound_n_rotate()
        # Fig. 1c,d; Fig. 2b, c
        analyze_dlc()

    # PREPROCESS Scorings

    if run_scoring_analysis:
        print("Running scoring analysis...")
        scoring_analysis()

    # RUN VAME
    if run_vame_model:
        print("Running VAME...")
        run_vame()

    # Dwell time (state-wise) ANALYSIS
    if run_dwell_time_analysis:
        print("Running dwell time analysis...")
        # Fig. 2e, f
        dwell_time()

    # Transition (between states) ANALYSIS
    if run_transition_matrix_analysis:
        print("Running transition matrix analysis...")
        # Fig. 3a, b, c, e
        # Supplementary Fig. 3a, b, c, d
        transition_matrix()

    # Latent representation (dynamics) ANALYSIS
    if run_latent_visualization:
        print("Running latent visualization...")
        # Fig.4 a, b,
        # Supplementary Fig. 4a,b
        latent_visualization()

        # Fig.4 c
        # Supplementary Fig. 4c
        compute_Wasserstein_distances()
        plot_Wasserstein_distance()

    if run_continuous_metric:
        print("Running continous analysis for metrics...")
        # Fig. 3d
        sliding_window()


    if run_classificaction:
        print("Running classification...")
        # Table 2
        # Supplementary Fig. 2c
        classifier2()

    if run_cv_motif_usage:
        print("Plotting CV model motif usage...")
        # Supplementary Fig. 2a, b
        motif_usage()

if __name__ == "__main__":
    main()
