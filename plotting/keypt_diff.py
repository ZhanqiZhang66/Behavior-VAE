# Created by Victoria Zhang at 2/12/2025
# File: keypt_diff.py.py
# Description: 
# Scenario:
# Usage:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

# %%
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


def plot_motif_comparison(data_pth, lb, ub, a, b):
    # Load the CSV file

    df = pd.read_csv(data_pth, index_col=0)
    df.index.name = "Keypoint Pair"

    # Function to extract data for given pairs
    def extract_data(pairs, motif):
        return np.array([df.loc[pair, f"Motif {motif}"] for pair in pairs])

    # Extract data for Lower Body
    lower_a_data = extract_data(lb, a)  # this should be a list of all data points, instead of the mean
    lower_b_data = extract_data(lb, b)

    # Extract data for Upper Body
    upper_a_data = extract_data(ub, a)
    upper_b_data = extract_data(ub, b)

    # Perform paired t-test for statistical significance
    lower_p_value = ttest_rel(lower_a_data, lower_b_data).pvalue
    upper_p_value = ttest_rel(upper_a_data, upper_b_data).pvalue

    lb_categories = np.tile([str(cat) for cat in lb], 2)

    colors = ['#1F77B4', '#FF7F0E']
    # Plotting Lower Body Rotation]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_lower = pd.DataFrame({
        "Motif": [a] * len(lower_a_data) + [b] * len(lower_b_data),
        "Average Distance": list(lower_a_data) + list(lower_b_data),
        "Category": lb_categories
    })

    ub_categories = np.tile([str(cat) for cat in ub], 2)
    df_upper = pd.DataFrame({
        "Motif": [a] * len(upper_a_data) + [b] * len(upper_b_data),
        "Average Distance": list(upper_a_data) + list(upper_b_data),
        "Category": ub_categories
    })

    df_lower = df_lower.dropna()
    df_upper = df_upper.dropna()

    sns.violinplot(data=df_upper, x="Category", y="Average Distance", hue="Motif",
                   split=True, inner="points", palette=[colors[0], colors[1]], ax=axes[0])

    axes[0].set_xticks(range(len(df_lower["Category"].unique())))  # Set numerical positions
    axes[0].set_xticklabels([str(cat) for cat in df_lower["Category"].unique()], rotation=45,
                            ha="right")  # Assign string labels

    axes[0].set_ylabel("Average Distance")
    axes[0].set_title(
        f"Lower Body Rotation: {MOTIF_DESCRIPTIONS[a]} vs {MOTIF_DESCRIPTIONS[b]}\n(p-value: {lower_p_value:.4f})")
    axes[0].legend(title="Motif")

    sns.violinplot(data=df_upper, x="Category", y="Average Distance", hue="Motif",
                   split=True, inner="quartile", palette=[colors[0], colors[1]], ax=axes[1])
    axes[1].set_xticks(range(len(df_upper["Category"].unique())))  # Set numerical positions
    axes[1].set_xticklabels([str(cat) for cat in df_upper["Category"].unique()], rotation=45,
                            ha="right")  # Assign string labels
    axes[1].set_ylabel("Average Distance")
    axes[1].set_title(
        f"Upper Body Rotation: {MOTIF_DESCRIPTIONS[a]} vs {MOTIF_DESCRIPTIONS[b]}\n(p-value: {lower_p_value:.4f})")
    axes[1].legend(title="Motif")

    fig.tight_layout()  # Ensures everything fits well
    fig.show()  # Display the full figure

    # width = 0.35
    #
    # x_lower = np.arange(len(lb))
    # axes[0].bar(x_lower - width / 2, lower_a_data, width, label=f"Motif {a}", yerr=lower_a_std, capsize=5,
    #             color=colors[0])
    # axes[0].bar(x_lower + width / 2, lower_b_data, width, label=f"Motif {b}", yerr=lower_b_std, capsize=5,
    #             color=colors[1])
    # axes[0].set_xticks(x_lower)
    # axes[0].set_xticklabels(lb, rotation=45, ha="right")
    # axes[0].set_ylabel("Average Distance")
    # axes[0].set_title(
    #     f"Lower Body Rotation: {MOTIF_DESCRIPTIONS[a]} vs {MOTIF_DESCRIPTIONS[b]}\n(p-value: {lower_p_value:.4f})")
    # axes[0].legend()

    # Plotting Upper Body/Head Rotation
    # x_upper = np.arange(len(ub))
    # axes[1].bar(x_upper - width / 2, upper_a_data, width, label=f"Motif {a}", yerr=upper_a_std, capsize=5,
    #             color=colors[0])
    # axes[1].bar(x_upper + width / 2, upper_b_data, width, label=f"Motif {b}", yerr=upper_b_std, capsize=5,
    #             color=colors[1])
    # axes[1].set_xticks(x_upper)
    # axes[1].set_xticklabels(ub, rotation=45, ha="right")
    # axes[1].set_ylabel("Average Distance")
    # axes[1].set_title(
    #     f"Upper Body/Head Rotation: {MOTIF_DESCRIPTIONS[a]} vs {MOTIF_DESCRIPTIONS[b]}\n(p-value: {upper_p_value:.4f})")
    # axes[1].legend()

    # Output statistical results
    print(f"Lower Body Rotation p-value: {lower_p_value:.4f}")
    print(f"Upper Body/Head Rotation p-value: {upper_p_value:.4f}")


# %%
from plotting.get_paths import get_my_path

# %%
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']
data_pth = fr"{github_path}\Behavior-VAE\data\all_pairwise_distances.csv"  # TODO
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

plot_motif_comparison(data_pth, lower_body_pairs, upper_body_pairs, 2, 7)
# %%
# Lower Body Pairs
lower_body_pairs = [

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

plot_motif_comparison(data_pth, lower_body_pairs, upper_body_pairs, 3, 4)
