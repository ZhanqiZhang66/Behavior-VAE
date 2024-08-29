# Characterizing Behavioral Dynamics in Bipolar Disorder with Computational Ethology

## Overview

This repository contains the code and analysis pipeline used for the preprint **"Characterizing Behavioral Dynamics in Bipolar Disorder with Computational Ethology."** The study leverages data-driven machine learning techniques to analyze natural, spontaneous open-field human behaviors in individuals with euthymic bipolar disorder (BD) and non-BD participants. The goal is to identify distinct behavioral motifs and propose novel metrics that reflect cognitive functions in BD, offering a new approach to characterizing neuropsychiatric conditions through behavior.

## Repository Structure

- **`main.py`**: The main script orchestrating the entire analysis pipeline. The code is modular, allowing the execution of specific sections by setting corresponding flags to `True`.
- **`preprocessing/`**: Contains scripts for preprocessing DLC markers and skeletons.
- **`scoring_analysis/`**: Scripts for analyzing and preprocessing scoring data.
- **`vame_model/`**: Scripts related to running and analyzing the VAME model for behavioral motif extraction.
- **`dwell_time_analysis/`**: Scripts for dwell time analysis.
- **`transition_matrix_analysis/`**: Scripts for transition matrix analysis.
- **`latent_visualization/`**: Scripts for visualizing latent representations of behavioral dynamics.
- **`continuous_metric_analysis/`**: Scripts for continuous metric analysis.
- **`classification/`**: Scripts for classification of BD and non-BD participants based on the computed behavioral features.
- **`cv_motif_usage/`**: Scripts for cross-validation and motif usage analysis.

## Running the Code

### Preprocessing DLC Markers and Skeletons

To preprocess DLC markers and skeletons, set `run_dlc_preprocessing = True` in `main.py`. This step includes the following:
- **Bounding and Rotating Data**: Align and prepare DLC marker data.
- **DLC Analysis**: Perform initial analysis to generate basic visualizations.

### Scoring Analysis

For scoring analysis, set `run_scoring_analysis = True`. This step includes:
- **Scoring Analysis**: Preprocess and analyze scoring data, crucial for subsequent steps.

### VAME Model Execution

To run the VAME model, set `run_vame_model = True`. This step extracts motifs representing distinct sets of actions from the behavioral data.

### Dwell Time Analysis

For state-wise dwell time analysis, set `run_dwell_time_analysis = True`. This step provides insights into the time spent in each behavioral state.

### Transition Matrix Analysis

To analyze transitions between behavioral states, set `run_transition_matrix_analysis = True`. This step involves the generation of transition matrices and associated visualizations.

### Latent Visualization

For visualizing latent representations, set `run_latent_visualization = True`. This step includes:
- **Wasserstein Distance Computation**: Quantify differences between distributions of behavioral states.
- **Plotting**: Generate visualizations of the latent space dynamics.

### Continuous Metric Analysis

For continuous metric analysis, set `run_continuous_metric = True`. This step uses sliding windows to compute and analyze metrics over time.

### Classification

To classify BD and non-BD participants, set `run_classificaction = True`. This step involves training and evaluating classifiers using the computed behavioral features.

### Cross-Validation and Motif Usage

To analyze cross-validation and motif usage, set `run_cv_motif_usage = True`. This step includes generating visualizations and metrics to assess model performance and motif utilization.

### Citation

If you use this code or the findings from this preprint, please cite the following:

Zhanqi Zhang, Chi K. Chou, Holden Rosberg, William Perry, Jared W Young, Arpi Minassian, Gal Mishne, Mikio Aoi, "Characterizing Behavioral Dynamics in Bipolar Disorder with Computational Ethology."


```latex
@article{zhang2024characterizing,
  title={Characterizing Behavioral Dynamics in Bipolar Disorder with Computational Ethology},
  author={Zhanqi Zhang, Chi K. Chou, Holden Rosberg, William Perry, Jared W Young, Arpi Minassian, Gal Mishne, Mikio Aoi},
  year={2024}
}