# Created by Victoria Zhang at 2/12/2025
# File: keypt_diff.py.py
# Description: 
# Scenario:
# Usage:

# %%

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind

from plotting.get_paths import get_my_path


# %%
def analyze_motifs(patients, keypoints, project_path, n_clusters=10, likelihood_threshold=0.9):
    # Define base paths
    results_path = os.path.join(project_path, "results")
    pose_estimation_path = os.path.join(project_path, "videos\pose_estimation")

    keypoint_pairs = [
        (keypoints[i], keypoints[j]) for i in range(len(keypoints)) for j in range(i)
    ]  # List of keypoint pairs for CSV
    motif_distances = {f"Motif {k}": [] for k in range(n_clusters)}

    all_pairwise_distances = []  # Store raw distance data

    for k in range(n_clusters):
        for patient in patients:
            # Paths to the VAME labels and pose estimation data
            vame_label_path = os.path.join(
                results_path,
                patient,
                "VAME",
                f"kmeans-{n_clusters}",
                f"{n_clusters}_km_label_{patient}.npy",
            )
            pose_data_path = os.path.join(pose_estimation_path, f"{patient}.csv")

            try:
                # Load cluster labels
                vame_label = np.load(vame_label_path)
            except FileNotFoundError:
                print(f"File not found: {vame_label_path}")
                continue

            try:
                # Load pose estimation data
                pose_data = pd.read_csv(pose_data_path, skiprows=[0], header=[0, 1])
            except FileNotFoundError:
                print(f"File not found: {pose_data_path}")
                continue
            print(patient)

            # Extract frames and drop the frame number column
            pose_data = pose_data.drop(columns=pose_data.columns[0])

            # Separate likelihood and coordinate data
            likelihood_columns = [
                col for col in pose_data.columns if col[1] == "likelihood"
            ]
            likelihood_data = pose_data[likelihood_columns]
            pose_data = pose_data.loc[
                        :, pose_data.columns.get_level_values(1) != "likelihood"
                        ]

            # Filter columns for target keypoints
            filtered_columns = [col for col in pose_data.columns if col[0] in keypoints]
            pose_data = pose_data[filtered_columns]
            likelihood_data = likelihood_data[
                [col for col in likelihood_columns if col[0] in keypoints]
            ]

            # Identify frames corresponding to the current cluster
            frame_indices = np.where(vame_label == k)[0]
            cluster_data = pose_data.iloc[frame_indices]
            cluster_likelihoods = likelihood_data.iloc[frame_indices]

            # Filter frames based on likelihood threshold
            valid_frames = (cluster_likelihoods >= likelihood_threshold).all(axis=1)
            cluster_data = cluster_data[valid_frames]

            if cluster_data.empty:
                continue

            # Compute pairwise distances for each frame
            for frame_id, frame in cluster_data.iterrows():
                keypoint_coords = frame.values.reshape(-1, 2)  # Reshape to (n_keypoints, 2)
                pairwise_distances = squareform(pdist(keypoint_coords))
                flattened_distances = pairwise_distances[np.tril_indices(len(keypoints), k=-1)]

                all_pairwise_distances.append([patient, frame_id, k] + list(flattened_distances))

    # Create DataFrame for all raw distances
    column_names = ["Patient", "Frame", "Motif"] + [" - ".join(pair) for pair in keypoint_pairs]
    pairwise_distances_df = pd.DataFrame(all_pairwise_distances, columns=column_names)

    # Save the full raw pairwise distances to CSV
    raw_csv_output_path = os.path.join(project_path, "raw_pairwise_distances.csv")
    pairwise_distances_df.to_csv(raw_csv_output_path, index=False)
    print(f"Raw pairwise distances saved to: {raw_csv_output_path}")

    # Optionally, compute mean distances and save to another CSV
    # Ensure numeric data for aggregation
    numeric_columns = pairwise_distances_df.drop(columns=["Patient", "Frame", "Motif"]).columns
    pairwise_distances_df[numeric_columns] = pairwise_distances_df[numeric_columns].apply(pd.to_numeric,
                                                                                          errors="coerce")

    # Compute mean distances per motif
    mean_distances_df = pairwise_distances_df.groupby("Motif")[numeric_columns].mean()

    mean_csv_output_path = os.path.join(project_path, "average_pairwise_distances.csv")
    mean_distances_df.to_csv(mean_csv_output_path, index_label="Motif")
    print(f"Average pairwise distances saved to: {mean_csv_output_path}")


# call the function


project_path = r"D:\OneDrive - UC San Diego\Data\Behavior_VAE_data\BD25-HC25-final-May17-2023"
results_path = os.path.join(project_path, "results")
patients = [os.path.splitext(filename)[0] for filename in os.listdir(results_path)]
patients.remove('classifier_result')
patients.remove('BC1JUPA')
patients.remove('BC1LUSE')
# Keypoints to analyze
keypoints = [
    "l-eye",
    "r-eye",
    "l_ear",
    "r_ear",
    "mouth",
    "center_neck",
    "l_sho",
    "r_sho",
    "l-elb",
    "r-elb",
    "l-hand",
    "r-hand",
    "center_hip",
    "l-hip",
    "r-hip",
    "l-knee",
    "r-knee",
    "l-feet",
    "r-feet",
    "center_feet",
]

analyze_motifs(
    patients, keypoints, project_path, n_clusters=10, likelihood_threshold=0.9
)
# %%
'''
PLOT
'''
# %%
from statsmodels.stats.multitest import multipletests

MOTIF_DESCRIPTIONS = {
    0: "torso rotation",
    1: "stretch (one body part)",
    2: "depart",
    3: "arm/hand movement",
    4: "static/fidget",
    5: "standstill",
    6: "wander and scan",
    7: "turn from/to",
    8: "examine/ with objects",
    9: "approach (aimed) or inspect"
}
project_name = 'BD25-HC25-final-May17-2023'
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']


# Updated function
def plot_motif_comparison(pairwise_distances_df, lb, ub, a, b):
    # Filter for motifs a and b
    pairwise_distances_df = pairwise_distances_df.copy()
    df_filtered = pairwise_distances_df[pairwise_distances_df["Motif"].isin([a, b])]
    df_filtered[lb + ub] = df_filtered[lb + ub].apply(pd.to_numeric, errors='coerce')

    # Extract the relevant keypoint pairs
    selected_columns = ["Motif"] + lb + ub  # Ensure these columns exist
    df_selected = df_filtered[selected_columns]

    # Melt DataFrame for plotting
    df_melted = df_selected.melt(id_vars=["Motif"], var_name="Keypoint Pair", value_name="Distance")

    # Perform paired t-test for statistical significance (Lower Body)
    lower_a_data = df_filtered[df_filtered["Motif"] == a][lb].values.flatten()
    lower_b_data = df_filtered[df_filtered["Motif"] == b][lb].values.flatten()

    # Perform paired t-test for statistical significance (Upper Body)
    upper_a_data = df_filtered[df_filtered["Motif"] == a][ub].values.flatten()
    upper_b_data = df_filtered[df_filtered["Motif"] == b][ub].values.flatten()

    # Compute p-values for each keypoint pair separately
    lower_p_values = {pair: ttest_ind(
        df_filtered[df_filtered["Motif"] == a][pair].dropna(),
        df_filtered[df_filtered["Motif"] == b][pair].dropna(),
        nan_policy='omit'
    ).pvalue for pair in lb}

    upper_p_values = {pair: ttest_ind(
        df_filtered[df_filtered["Motif"] == a][pair].dropna(),
        df_filtered[df_filtered["Motif"] == b][pair].dropna(),
        nan_policy='omit',
        equal_var=False
    ).pvalue for pair in ub}

    lower_p_values_raw = list(lower_p_values.values())
    upper_p_values_raw = list(upper_p_values.values())

    # Adjust p-values using Benjamini-Hochberg procedure
    lower_p_values_adjusted = multipletests(lower_p_values_raw, alpha=0.05, method='fdr_bh')[1]
    upper_p_values_adjusted = multipletests(upper_p_values_raw, alpha=0.05, method='fdr_bh')[1]

    # Convert back to dictionary with corresponding keypoint pairs
    lower_p_values_ = dict(zip(lb, lower_p_values_adjusted))
    upper_p_values_ = dict(zip(ub, upper_p_values_adjusted))

    print(lower_p_values_)

    for pair, pval in lower_p_values_.items():
        if pval == 0:
            print(f"{pair}: <1e-308")  # Display very small values as a lower bound
        else:
            print(f"{pair}: {pval:.{max(1, abs(int('{:.0e}'.format(pval).split('e')[-1])))}f}")
    print(upper_p_values_)
    for pair, pval in upper_p_values_.items():
        if pval == 0:
            print(f"{pair}: <1e-308")  # Display very small values as a lower bound
        else:
            print(f"{pair}: {pval:.{max(1, abs(int('{:.0e}'.format(pval).split('e')[-1])))}f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = plt.get_cmap('tab20')
    colors = {a: cmap(a * 2), b: cmap(b * 2)}
    # Lower Body
    sns.boxplot(data=df_melted[df_melted["Keypoint Pair"].isin(lb)], x="Keypoint Pair", y="Distance", hue="Motif",
                palette=colors, ax=axes[0])
    axes[0].set_title(
        f"Upper Body Rotation: {MOTIF_DESCRIPTIONS[a]} vs {MOTIF_DESCRIPTIONS[b]}\nLower Body Pairs\n" + "\n".join(
            [f"{pair}: p={pval:.6f}" for pair, pval in lower_p_values_.items()]))

    # Upper Body
    sns.boxplot(data=df_melted[df_melted["Keypoint Pair"].isin(ub)], x="Keypoint Pair", y="Distance", hue="Motif",
                palette=colors, ax=axes[1])
    axes[1].set_title(
        f"Upper Body Rotation: {MOTIF_DESCRIPTIONS[a]} vs {MOTIF_DESCRIPTIONS[b]}\nUpper Body Pairs\n" + "\n".join(
            [f"{pair}: p={pval:.6f}" for pair, pval in upper_p_values_.items()]))

    # Adjust layout and show
    plt.tight_layout()
    plt.show()
    sns.despine()
    fig.show()
    pwd = r'{}\Behavior_VAE_data\{}\figure\motif_difference'.format(onedrive_path, project_name)
    Path(pwd).mkdir(parents=True, exist_ok=True)
    fname = f"{a}-{b}-difference.png"
    fname_pdf = f"{a}-{b}-difference.pdf"
    fig.savefig(os.path.join(pwd, fname), transparent=True)
    fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)


# %%
df = pd.read_csv(
    r"D:\OneDrive - UC San Diego\Data\Behavior_VAE_data\BD25-HC25-final-May17-2023\raw_pairwise_distances.csv")
pairwise_distances_df = df
# %%
'''
# 1. Turning vs. Departing
Motif 2 (Depart from Previous Action) vs. Motif 7 (Turn From/To)
'''
# Lower Body Rotation keypoint pairs (correct order)
lower_body_pairs = [
    "l-hip - center_hip",
    "r-hip - center_hip",
    "r-hip - l-hip"
]

# Upper Body/Head Rotation keypoint pairs (correct order)
upper_body_pairs = [
    "l_sho - center_neck",
    "r_sho - center_neck",
    "center_neck - l-eye",
    "center_neck - r-eye"
]

# Call the function
# plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 2, 7)
plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 2, 7)

# %%
'''
# 2. Arm Movement vs. Fidgeting
Motif 3 (Arm and Hand Movement) vs. Motif 4 (Static or Fidget)
'''

# Lower Body Pairs
lower_body_pairs = [
    "l-hip - center_hip",
    "r-hip - center_hip",
    "r-hip - l-hip"
]

# Upper Body Pairs
upper_body_pairs = [
    "l-hand - l-elb",
    "r-hand - r-elb",
    "l-elb - l_sho",
    "r-elb - r_sho",
    "center_neck - r-eye",
    "center_neck - l-eye"
]
plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 3, 4)
# %%
'''
3. Stretching vs. Fidgeting
Motif 1 (Stretch One Body Part) vs. Motif 4 (Static or Fidget)
'''
limb_extension_pairs = [

    "l-feet - l-hip",
    "r-feet - r-hip"
]

# Head Stability Pair
head_stability_pair = [
    "center_hip - center_neck",
    "l-hand - l_sho",
    "r-hand - r_sho",
    "l-hand - l-elb",
    "r-hand - r-elb",
]
plot_motif_comparison(pairwise_distances_df, limb_extension_pairs, head_stability_pair, 1, 4)
# %%
'''
4. Arm Movement vs. Object Interaction
Motif 3 (Arm and Hand Movement) vs. Motif 8 (Examine/Interact with Objects)
'''
# Upper Body Pairs
upper_body_pairs = [
    "l-hand - l-elb",
    "r-hand - r-elb",
    "center_neck - l-eye",
    "center_neck - r-eye",
    "l-hand - center_neck",
    "r-hand - center_neck"
]

# Lower Body Pairs
lower_body_pairs = [

]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 3, 8)
# %%
'''
5. Wander vs. Purposeful Approach
Motif 6 (Wander and Scan Aimlessly) vs. Motif 9 (Approach with Aim and/or Inspect)
'''
# Lower Body Pairs
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip",
    "r-feet - l-feet"
]

# Upper Body Pairs
upper_body_pairs = [
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 6, 9)
# %%
'''
6. Static (Fidget) vs. Standstill
Motif 4 (Static or Fidget) vs. Motif 5 (Standstill)
'''
# Lower Body Pairs (Micro-movements)
lower_body_pairs = [
    "l-hand - l-elb",
    "r-hand - r-elb"
]

# Upper Body Pairs (Head/Eye Shifts)
upper_body_pairs = [
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 4, 5)
# %%
'''
Motif 1 (Stretch One Body Part) vs Motif 9 (Approach with Aim and/or Inspect)
'''
# Motif 1 vs Motif 9
lower_body_pairs_1_9 = [
    "l-hip - center_hip",
    "r-hip - center_hip",
    "l-feet - center_hip",
    "r-feet - center_hip"
]

upper_body_pairs_1_9 = [
    "l_sho - center_neck",
    "r_sho - center_neck",
    "center_neck - l-eye",
    "center_neck - r-eye",
    "l-hand - l-elb",
    "r-hand - r-elb"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs_1_9, upper_body_pairs_1_9, 1, 9)
# %%
'''
For Motif 4 (Static or Fidget) vs Motif 9 (Approach with Aim and/or Inspect)
'''
# Motif 4 vs Motif 9
lower_body_pairs_4_9 = [
    "l-feet - center_hip",
    "r-feet - center_hip"
]

upper_body_pairs_4_9 = [
    "l-hand - l-elb",
    "r-hand - r-elb",
    "l-elb - l_sho",
    "r-elb - r_sho",
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs_4_9, upper_body_pairs_4_9, 4, 9)
# %%
'''
Motif 5 (Standstill) vs. Motif 6 (Wander and Scan Aimlessly)
Motif 5 has minimal to no movement across all keypoints.
Motif 6 involves continuous, aimless movement of both the lower body (feet, hips) and head/eyes for scanning.
'''
# Motif 5 vs Motif 6
lower_body_pairs_5_6 = [
    "l-feet - center_hip",
    "r-feet - center_hip",
    "l-hip - center_hip",
    "r-hip - center_hip"
]

upper_body_pairs_5_6 = [
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs_5_6, upper_body_pairs_5_6, 5, 6)
# %%
'''
Motif 3 (Arm and Hand Movement) vs. Motif 9 (Approach with Aim and/or Inspect)
Why Different:

Motif 3 focuses mainly on upper body movements (arms, hands) with limited lower body movement.
Motif 9 requires full-body engagement, involving both purposeful stepping and focused head/eye movement.
## Keypoints to Compare:

### Lower Body (for approach):
- center_hip - l-feet
- center_hip - r-feet
### Upper Body (for reaching):
- l-hand - l-elb
- r-hand - r-elb
- center_neck - l_eye
- center_neck - r_eye
'''
# Motif 3 vs Motif 9
lower_body_pairs_3_9 = [
    "l-feet - center_hip",
    "r-feet - center_hip"
]

upper_body_pairs_3_9 = [
    "l-hand - l-elb",
    "r-hand - r-elb",
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs_3_9, upper_body_pairs_3_9, 3, 9)
# %% md
'''
Motif 1 (Stretch One Body Part) vs. Motif 7 (Turn From/To)
Why Different:

Motif 1 involves slow, deliberate limb extension without much rotational motion.
Motif 7 requires sharp, rotational movements of both the upper body and lower body (especially hips and feet).
## Keypoints to Compare:

### Lower Body Rotation:
- center_hip - l-hip
- center_hip - r-hip
- l-hip - r-hip
### Upper Body Rotation vs. Stretch:
- center_neck - l_sho
- center_neck - r_sho
'''
# Motif 1 vs Motif 7
lower_body_pairs_1_7 = [
    "l-hip - center_hip",
    "l-hip - center_hip",
    "r-hip - l-hip"
]
upper_body_pairs_1_7 = [
    "l_sho - center_neck",
    "r_sho - center_neck"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs_1_7, upper_body_pairs_1_7, 1, 7)
# %%
'''
Motif 4 (Static or Fidget) vs. Motif 6 (Wander and Scan Aimlessly)
Why Different:

Motif 4 involves tiny, localized micro-movements of hands/fingers.
Motif 6 involves broad, whole-body movement, especially legs/feet and head/eyes.
## Keypoints to Compare:

### Lower Body for Walking:
- l-feet - center_hip
- r-feet - center_hip
### Micro-Movements in Fidgeting:
- l-hand - l-elb
- r-hand - r-elb
- center_neck - l_eye
- center_neck - r_eye
'''
# Motif 4 vs Motif 6
lower_body_pairs_4_6 = [
    "l-feet - center_hip",
    "r-feet - center_hip"
]

upper_body_pairs_4_6 = [
    "l-hand - l-elb",
    "r-hand - r-elb",
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs_4_6, upper_body_pairs_4_6, 4, 6)

# %%
'''
# Motif 2 (Depart from Previous Action) vs. Motif 5 (Standstill)
Why this comparison?
Departing requires coordinated whole-body movement (especially in the lower body), while standstill should show minimal movement.

### Keypoints to Compare:

Lower Body: center_hip - l-feet, center_hip - r-feet, l-hip - r-hip
Upper Body: center_neck - l_sho, center_neck - r_sho
Head/Gaze: center_neck - l_eye, center_neck - r_eye
'''
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip",
    "r-hip - l-hip"
]

# Upper Body Pairs
upper_body_pairs = [
    "l_sho - center_neck",
    "r_sho - center_neck",
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 2, 5)
# %%
'''
Motif 0 (Torso Rotation) vs. Motif 3 (Arm and Hand Movement)
Why this comparison?
Torso rotation primarily engages the core and shoulders, while arm and hand movements focus on the upper limbs. These should show very different movement patterns.

## Keypoints to Compare:

Torso: center_hip - center_neck, center_neck - l_sho, center_neck - r_sho
Arms: l-hand - l-elb, r-hand - r-elb
'''
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip",
    "r-hip - l-hip"
]
# Upper Body Keypoint Pairs
upper_body_pairs = [
    "center_hip - center_neck",
    "l_sho - center_neck",
    "r_sho - center_neck",
    "l-hand - l-elb",
    "r-hand - r-elb"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 0, 3)
# %%
'''
# Motif 6 (Wander and Scan Aimlessly) vs. Motif 9 (Approach with Aim and/or Inspect)
Why this comparison?
Both involve movement, but wandering is more random, while approaching is purposeful and directed.

## Keypoints to Compare:

Lower Body: center_hip - l-feet, center_hip - r-feet, l-hip - r-hip
Head/Gaze: center_neck - l_eye, center_neck - r_eye
'''
# Lower Body Keypoint Pairs
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip",
    "r-hip - l-hip"
]

# Head/Gaze Keypoint Pairs
head_gaze_pairs = [
    "center_neck - r-eye",
    "center_neck - l-eye",
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, head_gaze_pairs, 6, 9)
# %%
'''
# Motif 1 (Stretch One Body Part) vs. Motif 8 (Examine/Interact with Objects)
Why this comparison?
Stretching involves large, slow, deliberate limb movements, while interacting involves precise hand movements with possible head/eye coordination.

## Keypoints to Compare:

Arms/Hands: l-hand - l-sho, r-hand - r-sho, l-hand - l-elb, r-hand - r-elb
Head/Gaze: center_neck - l_eye, center_neck - r_eye
'''

# Upper Body Keypoint Pairs (Arms/Hands and Head/Gaze)
upper_body_pairs = [
    "l-hand - l_sho",
    "r-hand - r_sho",
    "l-hand - l-elb",
    "r-hand - r-elb",
    "center_neck - l-eye",
    "center_neck - r-eye"
]

# Lower Body Keypoint Pairs (None specified for this case)
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip",
    "r-hip - l-hip"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 1, 8)
# %%
'''
# Motif 4 (Static or Fidget) vs. Motif 6 (Wander and Scan Aimlessly)
Why this comparison?
Fidgeting is small, localized movement, while wandering involves whole-body, spatial movement.

## Keypoints to Compare:

Lower Body: center_hip - l-feet, center_hip - r-feet
Hands: l-hand - l-elb, r-hand - r-elb
'''

# Upper Body Keypoint Pairs (Hands)
upper_body_pairs = [
    "l-hand - l-elb",
    "r-hand - r-elb"
]

# Lower Body Keypoint Pairs
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 4, 6)
# %%
'''
# Motif 0 (Torso Rotation) vs. Motif 7 (Turn From/To)
Why this comparison?
Torso rotation is an isolated movement of the upper body, whereas turning involves coordinated whole-body rotation.

## Keypoints to Compare:

Lower Body: center_hip - l-hip, center_hip - r-hip
Upper Body: center_neck - l_sho, center_neck - r_sho
Head: center_neck - l_eye, center_neck - r_eye
'''

# Lower Body Keypoint Pairs
lower_body_pairs = [
    "l-hip - center_hip",
    "r-hip - center_hip"
]

# Upper Body Keypoint Pairs
upper_body_pairs = [
    "l_sho - center_neck",
    "r_sho - center_neck",
    "center_neck - l-eye",
    "center_neck - r-eye"
]

plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 0, 7)
# %%
'''
# Motif 3 (Arm and Hand Movement) vs. Motif 5 (Standstill)
Why this comparison?
Arm and hand movements involve dynamic upper limb activity, while standstill should show minimal to no movement.

## Keypoints to Compare:

Arms: l-hand - l-elb, r-hand - r-elb, l-elb - l-sho, r-elb - r-sho
Head/Gaze: center_neck - l_eye, center_neck - r_eye
'''
# Upper Body (Arms) Keypoint Pairs
upper_body_pairs = [
    "l-hand - l-elb",
    "r-hand - r-elb",
    "l-elb - l_sho",
    "r-elb - r_sho",
    "center_neck - l-eye",
    "center_neck - r-eye"
]
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip",
    "r-hip - l-hip"
]
plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 3, 5)
# %%
'''
# Motif 8 (Examine/Interact with Objects) vs. Motif 9 (Approach with Aim and/or Inspect)
Why this comparison?
Motif 8 focuses on precise hand-object interaction, while Motif 9 involves broader whole-body movement toward a target.
'''

# Lower Body Keypoint Pairs
lower_body_pairs = [
    "l-feet - center_hip",
    "r-feet - center_hip"
]

# Hands/Arms Keypoint Pairs
upper_body_pairs = [
    "l-hand - l-elb",
    "r-hand - r-elb",
    "center_neck - l-eye",
    "center_neck - r-eye"
]
plot_motif_comparison(pairwise_distances_df, lower_body_pairs, upper_body_pairs, 8, 9)
