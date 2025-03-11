# %%
import itertools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from scipy.stats import f_oneway, tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from data.load_data import load_pt_data

# %% Path
if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
    data_path = rf"C:\Users\zhanq\OneDrive - UC San Diego\SURF"
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC' or 'Victoria-Dell':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
    onedrive_path = r'D:\OneDrive - UC San Diego'
    data_path = rf"D:\OneDrive - UC San Diego\SURF"
else:
    github_path = r'C:\Users\kiet\OneDrive - UC San Diego\GitHub'
    data_path = rf"C:\Users\kietc\OneDrive - UC San Diego\SURF"
# %% Data Path
random.seed(42)
data, YMRS, HAM_D, start_frame, condition, isBD = load_pt_data(
    video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD = [k for k, v in isBD.items() if v[0] == 'Euthymic']
videos = control_videos + BD

random_seed = 14
diagnostic_path = rf"{data_path}\scaled_diagnostic_data.csv"
assessment_score_path = rf"{data_path}\assessment_scores_seed_{random_seed}"

vame_motif_path = rf"{data_path}\VAME\motif_usage_overall.csv"
vame_motif_epoch_path = rf"{data_path}\VAME\motif_usage_3_split.csv"
vame_ens_path = rf'{data_path}\VAME\ens_3_split.csv'
# ens per motif
vame_ensm_path = rf'{data_path}\VAME\ens_per_motif_3_split.csv'
vame_entropy_path = rf'{data_path}\VAME\entropy_3_split.csv'
vame_count_path = rf'{data_path}\VAME\count_3_split.csv'
vame_volume_path = rf"{data_path}\VAME\volume.csv"
vame_volume_per_motif_path = rf'{data_path}\VAME\volume_per_motif.csv'
vame_feature_selection_path = rf"{data_path}\VAME\feature_selection.csv"
vame_score_path = rf"{data_path}\VAME\scores_seed_{random_seed}"
vame_intradis_path = rf"{data_path}\VAME\interpopulation_distances.csv"
vame_interdis_path = rf"{data_path}\VAME\interpopulation_distances.csv"

dlc_motif_path = rf"{data_path}\DLC\motif_usage_overall.csv"
dlc_ens_path = rf'{data_path}\DLC\ens_3_split.csv'
# ens per motif
dlc_ensm_path = rf'{data_path}\DLC\ens_per_motif_3_split.csv'
dlc_entropy_path = rf'{data_path}\DLC\entropy_3_split.csv'
dlc_score_path = rf"{data_path}\DLC\scores_seed_{random_seed}"

hbpm_motif_path = rf"{data_path}\hBPM\motif_usage_overall.csv"
hbpm_ens_path = rf'{data_path}\hBPM\ens_3_split.csv'
# ens per motif
hbpm_ensm_path = rf'{data_path}\hBPM\ens_per_motif_3_split.csv'
hbpm_entropy_path = rf'{data_path}\hBPM\entropy_3_split.csv'
hbpm_count_path = rf'{data_path}\hBPM\count_3_split.csv'
hbpm_score_path = rf"{data_path}\hBPM\scores_seed_{random_seed}"

s3d_motif_path = rf"{data_path}\S3D\motif_usage_overall.csv"
s3d_ens_path = rf'{data_path}\S3D\ens_3_split.csv'
# ens per motif
s3d_ensm_path = rf'{data_path}\S3D\ens_per_motif_3_split.csv'
s3d_entropy_path = rf'{data_path}\S3D\entropy_3_split.csv'
s3d_count_path = rf'{data_path}\S3D\count_3_split.csv'
s3d_score_path = rf"{data_path}\S3D\scores_seed_{random_seed}"

mmaction_motif_path = rf"{data_path}\MMAction\motif_usage_overall.csv"
mmaction_ens_path = rf'{data_path}\MMAction\ens_3_split.csv'
# ens per motif
mmaction_ensm_path = rf'{data_path}\MMAction\ens_per_motif_3_split.csv'
mmaction_entropy_path = rf'{data_path}\MMAction\entropy_3_split.csv'
mmaction_count_path = rf'{data_path}\MMAction\count_3_split.csv'
mmaction_score_path = rf"{data_path}\MMAction\scores_seed_{random_seed}"

export_path = rf"{data_path}\Classification\{random_seed}"
export_result_path = rf"{data_path}\Classification\{random_seed}"

medication_path = f"{onedrive_path}\Data\Behavior_VAE_data\medication_list.csv"

ensm_epoch3 = ['ens_epoch3_m0', 'ens_epoch3_m1', 'ens_epoch3_m2', 'ens_epoch3_m3', 'ens_epoch3_m4',
               'ens_epoch3_m5', 'ens_epoch3_m6', 'ens_epoch3_m7', 'ens_epoch3_m8', 'ens_epoch3_m9']
ensm_epoch1 = ['ens_epoch1_m0', 'ens_epoch1_m1', 'ens_epoch1_m2', 'ens_epoch1_m3', 'ens_epoch1_m4',
               'ens_epoch1_m5', 'ens_epoch1_m6', 'ens_epoch1_m7', 'ens_epoch1_m8', 'ens_epoch1_m9']
ensm_diff = ['ens_diff_m0', 'ens_diff_m1', 'ens_diff_m2', 'ens_diff_m3', 'ens_diff_m4',
             'ens_diff_m5', 'ens_diff_m6', 'ens_diff_m7', 'ens_diff_m8', 'ens_diff_m9']
n_subject_in_population = 24
data, YMRS, HAM_D, start_frame, condition, isBD = load_pt_data(
    video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']

YMRS_score = []
HAM_D_score = []
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        YMRS_score.append(YMRS[v])
        HAM_D_score.append(HAM_D[v])

volume_per_motif_epoch3 = ['vol_epoch3_motif0', 'vol_epoch3_motif1', 'vol_epoch3_motif2', 'vol_epoch3_motif3',
                           'vol_epoch3_motif4', 'vol_epoch3_motif5', 'vol_epoch3_motif6',
                           'vol_epoch3_motif7', 'vol_epoch3_motif8', 'vol_epoch3_motif9']
volume_per_motif_epoch1 = ['vol_epoch2_motif0', 'vol_epoch2_motif1',
                           'vol_epoch2_motif2', 'vol_epoch2_motif3', 'vol_epoch2_motif4',
                           'vol_epoch2_motif5', 'vol_epoch2_motif6', 'vol_epoch2_motif7',
                           'vol_epoch2_motif8', 'vol_epoch2_motif9', ]
volume_per_motif_diff = ['vol_diff_motif0', 'vol_diff_motif1', 'vol_diff_motif2', 'vol_diff_motif3',
                         'vol_diff_motif4', 'vol_diff_motif5', 'vol_diff_motif6', 'vol_diff_motif7',
                         'vol_diff_motif8', 'vol_diff_motif9']


def classify(df, features, max_iter, seed):
    X = df[features]
    y = df['BD']
    # print(X.columns)

    acc = []
    pre = []
    rec = []

    for i in range(seed, seed + max_iter):
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.24, stratify=y, random_state=i)

        # model
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(xtrain, ytrain)
        y_pred = classifier.predict(xtest)

        acc.append(accuracy_score(ytest, y_pred))
        pre.append(precision_score(ytest, y_pred))
        rec.append(recall_score(ytest, y_pred))

        # cross validation
        scoring = ['accuracy', 'precision', 'recall']
        scores = cross_validate(classifier, xtrain, ytrain, scoring=scoring, cv=3)

        acc.extend(scores['test_accuracy'])
        pre.extend(scores['test_precision'])
        rec.extend(scores['test_recall'])

    print('Accuracy: %.05f (%.05f)' % (np.mean(acc), np.std(acc)))
    print('Precision: %.05f (%.05f)' % (np.mean(pre), np.std(pre)))
    print('Recall: %.05f (%.05f)' % (np.mean(rec), np.std(rec)))
    return [acc, pre, rec]


def feature_selection(X, y, tol=-0.02):
    lgr = linear_model.LogisticRegression()
    sfs = SequentialFeatureSelector(lgr,
                                    n_features_to_select="auto",
                                    tol=tol,  # a hyperparameter that gives < half of all features
                                    direction="backward",
                                    scoring='accuracy',
                                    cv=4,
                                    n_jobs=5)
    selector = sfs.fit(X, y)
    selected_features_list = selector.get_support()
    selected_features = list(X.columns[selected_features_list])
    print(f"selected: {selected_features}")
    print(f"selected {len(selected_features)} out of {np.shape(X)[1]} features")
    return selected_features


def drop_rows_if_50(df):
    # Remove unqualified person rows from the csv,
    # which are the 27th and 36th rows JUPA and LUSE
    # if the length of the dataframe is 50
    if len(df) == 50:
        df.drop(index=[27, 36], inplace=True)
    return df


# %% Reading medication data

medication_df = pd.read_csv(medication_path)
medication_df = drop_rows_if_50(medication_df)
medication_df['video'] = medication_df['subject_name']
medication_df.drop('subject_name', axis=1, inplace=True)

# %% Reading assessment data

assessment_df = pd.read_csv(diagnostic_path)
assessment_df.drop('gender', axis=1, inplace=True)

bd_df = assessment_df[['video', 'BD']]
assessment_df.drop('video', axis=1, inplace=True)
assessment_df = drop_rows_if_50(assessment_df)
# %% Reading VAME data


# %%
# Motif dwell time
vame_motif_df = pd.read_csv(vame_motif_path)
vame_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

vame_motif_epoch_df = pd.read_csv(vame_motif_epoch_path)

# ENS(transition matrix)
vame_ens_df = pd.read_csv(vame_ens_path)
vame_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# ENS(each row of motif in transition matrix)
vame_ensm_df = pd.read_csv(vame_ensm_path)
vame_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0]) + 1}_m{x[6:]}' if 'motif' in x else x, inplace=True)

# Entropy of transition matrix
vame_entropy_df = pd.read_csv(vame_entropy_path)
vame_entropy_df.rename(columns=lambda x: f'entropy_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# Count of zeros in transition matrix
vame_count_df = pd.read_csv(vame_count_path)
vame_count_df.rename(columns=lambda x: f'cnt_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# Volume of each latent vector
vame_volume_df = pd.read_csv(vame_volume_path)
vame_volume_df.rename(columns=lambda x: f'vol_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# inter and intra population distance
vame_intradis_df = pd.read_csv(vame_intradis_path)
vame_interdis_df = pd.read_csv(vame_interdis_path)

vame_volume_per_motif_df = pd.read_csv(vame_volume_per_motif_path)

vame_volume_per_motif_df.rename(columns=lambda x: f'vol_epoch{int(x[0]) + 1}_motif{(x[6:])}' if 'motif' in x else x,
                                inplace=True)

vame_df = pd.merge(bd_df, vame_motif_df, on='video')
vame_df = pd.merge(vame_df, vame_motif_epoch_df, on='video')
vame_df = pd.merge(vame_df, vame_ens_df, on='video')
vame_df = pd.merge(vame_df, vame_ensm_df, on='video')
# vame_df = pd.merge(vame_df, vame_entropy_df, on='video')


vame_df = pd.merge(vame_df, vame_count_df, on='video')
vame_df = pd.merge(vame_df, vame_volume_df, on='video')

vame_df = pd.merge(vame_df, vame_volume_per_motif_df, on='video')

vame_df['ens_diff'] = vame_df['ens_epoch3'] - vame_df['ens_epoch1']
# vame_df['entropy_diff'] = vame_df['entropy_epoch3'] - vame_df['entropy_epoch1']
vame_df['vol_diff_2_1'] = vame_df['vol_epoch3'] - vame_df['vol_epoch2']
vame_df['vol_diff_1_0'] = vame_df['vol_epoch2'] - vame_df['vol_epoch1']
vame_df['vol_diff_2_0'] = vame_df['vol_epoch3'] - vame_df['vol_epoch1']

for i, v in enumerate(ensm_diff):
    vame_df[v] = vame_df[ensm_epoch3[i]] - vame_df[ensm_epoch1[i]]

vame_df = drop_rows_if_50(vame_df)

# %% Medication Effect table
'''
Medication Effect Analysis
'''
# Identify BD subjects who are NOT on any of the top 5 medications
top_5_normalized_meds = ['lictal', 'abilify', 'seroquel', 'lithium', 'zoloft']
bd_no_meds = medication_df[
    (medication_df[top_5_normalized_meds].sum(axis=1) == 0) & (medication_df["is_BD"] != "HC")]
hc_group = medication_df[medication_df["is_BD"] == "HC"]

bd_no_meds_data = vame_df[
    vame_df['video'].isin(['BC1LESA', 'BC1LOMI', 'BC1BRBU', 'BC1LUOR', 'BC1MEMA', 'BC1KEMA', 'BC1MISE', 'BC1CISI'])]

hc_all_data = vame_df[vame_df["video"].isin(hc_group["video"])]
bd_all_data = vame_df[vame_df["BD"] == 1]
bd_all_meds_data = bd_all_data[~
bd_all_data['video'].isin(['BC1LESA', 'BC1LOMI', 'BC1BRBU', 'BC1LUOR', 'BC1MEMA', 'BC1KEMA', 'BC1MISE', 'BC1CISI'])]

# Create a dictionary to store DataFrames for each medication
bd_meds_data = {}

# Iterate over the top 5 most common medications
for med in top_5_normalized_meds:
    # Select BD subjects who are taking this medication
    bd_meds_data[med] = bd_all_data[bd_all_data["video"].isin(medication_df[medication_df[med] == 1]["video"])].copy()
# %%
from statsmodels.stats.multitest import multipletests
# Initialize dictionary to store test results
test_results = {}

# List of BD medication groups to compare
bd_med_groups = {
    "lictal": bd_meds_data["lictal"],
    "abilify": bd_meds_data["abilify"],
    "seroquel": bd_meds_data["seroquel"],
    "lithium": bd_meds_data["lithium"],
    "zoloft": bd_meds_data["zoloft"],
}

# Iterate through each feature (excluding the first two columns)
for feature in bd_all_data.columns[2:]:
    # Extract values for each group
    bd_all_values = bd_all_data[feature].dropna()
    bd_all_meds_values = bd_all_meds_data[feature].dropna()
    bd_no_meds_values = bd_no_meds_data[feature].dropna()
    hc_values = hc_all_data[feature].dropna()

    # Perform statistical tests
    p_values = {
        "p_bd_vs_hc": ttest_ind(bd_all_values, hc_values, equal_var=False)[1] if len(bd_all_values) > 1 and len(
            hc_values) > 1 else None,
        "p_bd_no_meds_vs_hc": ttest_ind(bd_no_meds_values, hc_values, equal_var=False)[1] if len(
            bd_no_meds_values) > 1 and len(hc_values) > 1 else None,
        "p_bd_no_meds_vs_bd_meds": ttest_ind(bd_no_meds_values, bd_all_meds_values, equal_var=False)[1] if len(
            bd_no_meds_values) > 1 and len(bd_all_meds_values) > 1 else None,
    }

    # Filter out None values
    valid_p_values = {key: p for key, p in p_values.items() if p is not None}

    # Apply Benjamini-Hochberg correction
    if valid_p_values:
        _, p_corrected, _, _ = multipletests(list(valid_p_values.values()), method='fdr_bh')

        # Update p_values with corrected values
        corrected_p_values = dict(zip(valid_p_values.keys(), p_corrected))
        p_values.update(corrected_p_values)



    # # Perform tests for each BD medication group
    # for med, med_df in bd_med_groups.items():
    #     med_values = med_df[feature].dropna()
    #     p_values[f"p_{med}_vs_hc"] = ttest_ind(med_values, hc_values, equal_var=False)[1] if len(
    #         med_values) > 1 and len(hc_values) > 1 else None
    #     p_values[f"p_{med}_vs_bd_all"] = ttest_ind(med_values, bd_all_values, equal_var=False)[1] if len(
    #         med_values) > 1 and len(bd_all_values) > 1 else None

    # Store results
    test_results[feature] = p_values
#%%
# Convert results to DataFrame
test_results_df = pd.DataFrame.from_dict(test_results, orient="index")
test_results_df = test_results_df.round(3)
save_path = rf'{onedrive_path}\Data\Behavior_VAE_data\medication_effect_stat_test_BH.csv'
test_results_df.to_csv(save_path)

# Count how many times values in other columns are < 0.05 when p_bd_vs_hc is also < 0.05
significant_counts = (test_results_df.loc[test_results_df["p_bd_vs_hc"] < 0.05] < 0.05).sum()
# %%
'''
Age as a covariate analysis
'''

import pandas as pd
import statsmodels.formula.api as smf

# Load demographic data
file_path = rf'{onedrive_path}\Data\Behavior_VAE_data\demographic_table.csv'  # Adjust file path if needed
demographic_data = pd.read_csv(file_path)

# Filter out unwanted conditions
demographic_data = demographic_data[~demographic_data['condition'].isin(['JUPA', 'LUSE'])]

# Normalize BD condition labels
demographic_data['condition'] = demographic_data['condition'].replace(
    ['BD', 'BD1', 'BD2', 'Cyclothymic'], 'BD'
)
# Select only relevant columns (assuming subject ID exists for merging)
demographic_data = demographic_data[['video_name', 'condition', 'age']]
# Rename 'subject_id' to match behavioral data's 'video' column
demographic_data = demographic_data.rename(columns={'video_name': 'video'})

# Load behavioral data
behavioral_data = vame_df
# Merge with demographic data on subject_id
df = behavioral_data.merge(demographic_data, on='video')

df = df.rename(columns=lambda x: f"f_{x}" if x[0].isdigit() else x)
# Define the list of behavioral features to analyze
test_results_df = test_results_df.rename(index=lambda x: f"f_{x}" if x[0].isdigit() else x)
features = test_results_df[test_results_df['p_bd_vs_hc'] < 0.05].index.tolist()

# Run ANCOVA for each feature
results = {}
count = 0
count_mar = 0
for feature in features:
    full_model = smf.ols(f"{feature} ~ BD + age", data=df).fit()

    # Reduced model: only age
    reduced_model = smf.ols(f"{feature} ~ age", data=df).fit()

    # Compute effect size (Cohen’s f²)
    R2_full = full_model.rsquared
    R2_reduced = reduced_model.rsquared
    effect_size = (R2_full - R2_reduced) / (1 - R2_full) if R2_full < 1 else float('nan')

    # Extract key statistics
    coef_table = full_model.summary().tables[1].data  # Extract coefficient table
    headers = coef_table[0]  # Column headers
    values = {row[0]: row[1:] for row in coef_table[1:]}  # Extract values

    # Print only the relevant stats for 'condition' and 'age'
    print(f"\nFeature: {feature}")
    print(f"{headers[0]:<10} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<10}")
    print("-" * 60)
    for term in ["BD", "age"]:
        if term in values:
            print(
                f"{term:<10} | {values[term][0]:<10} | {values[term][1]:<10} | {values[term][2]:<10} | {values[term][3]:<10}")

    # Print a quick interpretation
    p_condition = float(values["BD"][3])  # Extract p-value for condition
    print(f"Effect Size (Cohen’s f²): {effect_size:.4f}")
    if p_condition < 0.05:
        print("✅ Condition remains significant after controlling for age.")
        count += 1
    elif p_condition < 0.06:
        print("⚠️Condition is marginally significant after controlling for age.")
        count_mar += 1
    else:
        print("❌ Condition is no longer significant after controlling for age.")

# # Store results in a text file if needed
# with open("ANCOVA_results.txt", "w") as f:
#     for feature, summary in results.items():
#         f.write(f"Feature: {feature}\n")
#         f.write(str(summary) + "\n\n")

# %%
vame_df.drop('video', axis=1, inplace=True)
# %% Wasserstein Distance will introduce NaNs that is not allowed for feature selector
# data, YMRS, HAM_D, start_frame, condition, isBD = load_pt_data(video_information_pth=
# r'{}\Behavior-VAE\data\video-information.csv'.format(
#     github_path))
# control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
# BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
# vame_interdis_df['video'] = control_videos + BD_videos
# vame_intradis_df['video'] = control_videos + BD_videos
#
# vame_intradis_df = vame_intradis_df.drop(columns=['BD'])
#
# vame_interdis_df = vame_interdis_df.drop(columns=['BD'])
#
# vame_df = pd.merge(vame_df, vame_interdis_df, on='video')
# vame_df = pd.merge(vame_df, vame_intradis_df, on='video')
# vame_df.drop('video', axis=1, inplace=True)
# for i,v in enumerate(volume_per_motif_diff):
#     vame_df[v] = vame_df[volume_per_motif_epoch3[i]] - vame_df[volume_per_motif_epoch1[i]]

# %%
"""
DLC
"""
# %%
dlc_motif_df = pd.read_csv(dlc_motif_path)
dlc_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS average between motif
dlc_ens_df = pd.read_csv(dlc_ens_path)
dlc_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# ENS per motif
dlc_ensm_df = pd.read_csv(dlc_ensm_path)
dlc_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0]) + 1}_m{x[6:]}' if 'motif' in x else x, inplace=True)

# Entropy
# dlc_entropy_df = pd.read_csv(dlc_entropy_path)
# dlc_entropy_df.rename(columns=lambda x: f'entropy_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

dlc_df = pd.merge(bd_df, dlc_motif_df, on='video')
dlc_df = pd.merge(dlc_df, dlc_ens_df, on='video')
dlc_df = pd.merge(dlc_df, dlc_ensm_df, on='video')
# dlc_df = pd.merge(dlc_df, dlc_entropy_df, on='video')
dlc_df.drop('video', axis=1, inplace=True)

dlc_df['ens_diff'] = dlc_df['ens_epoch3'] - dlc_df['ens_epoch1']
# dlc_df['entropy_diff'] = dlc_df['entropy_epoch3'] - dlc_df['entropy_epoch1']

for i, v in enumerate(ensm_diff):
    dlc_df[v] = dlc_df[ensm_epoch3[i]] - dlc_df[ensm_epoch1[i]]

# %%
"""
HBPM
"""
# %%
hbpm_motif_df = pd.read_csv(hbpm_motif_path)
hbpm_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)
# Ignore the first column
hbpm_motif_df_values = hbpm_motif_df.iloc[:, 1:].values
# Normalize by row
row_sums = hbpm_motif_df_values.sum(axis=1)
hbpm_motif_df1 = hbpm_motif_df_values / row_sums[:, np.newaxis]
# Combine with the first column
hbpm_motif_df = pd.DataFrame(np.hstack((hbpm_motif_df[['video']].values, hbpm_motif_df1)),
                             columns=hbpm_motif_df.columns)

# ENS average between motif
hbpm_ens_df = pd.read_csv(hbpm_ens_path)
hbpm_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# ENS per motif
hbpm_ensm_df = pd.read_csv(hbpm_ensm_path)
hbpm_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0]) + 1}_m{x[6:]}' if 'motif' in x else x, inplace=True)

# Entropy
# hbpm_entropy_df = pd.read_csv(hbpm_entropy_path)
# hbpm_entropy_df.rename(columns=lambda x: f'entropy_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

hbpm_df = pd.merge(bd_df, hbpm_motif_df, on='video')
hbpm_df = pd.merge(hbpm_df, hbpm_ens_df, on='video')
hbpm_df = pd.merge(hbpm_df, hbpm_ensm_df, on='video')
# hbpm_df = pd.merge(hbpm_df, hbpm_entropy_df, on='video')
hbpm_df.drop('video', axis=1, inplace=True)

hbpm_df['ens_diff'] = hbpm_df['ens_epoch3'] - hbpm_df['ens_epoch1']
# hbpm_df['entropy_diff'] = hbpm_df['entropy_epoch3'] - hbpm_df['entropy_epoch1']

hbpm_ensm_epoch3 = []
hbpm_ensm_epoch1 = []
hbpm_ensm_diff = []
for i in range(11):
    hbpm_ensm_epoch3.append('ens_epoch3_m' + str(i))
    hbpm_ensm_epoch1.append('ens_epoch3_m' + str(i))
    hbpm_ensm_diff.append('ens_epoch3_m' + str(i))

for i, v in enumerate(ensm_diff):
    hbpm_df[v] = hbpm_df[ensm_epoch3[i]] - hbpm_df[ensm_epoch1[i]]

# %%
"""
Spatial D
"""
spatialD_path = rf"{onedrive_path}\SURF\spatialD.csv"
spatialD_df = pd.read_csv(spatialD_path)
spatialD_df = pd.merge(bd_df, spatialD_df, on='video')
spatialD_df.drop('video', axis=1, inplace=True)
# %%
"""
S3D
"""
# %%
s3d_motif_df = pd.read_csv(s3d_motif_path)
s3d_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)
top_10_motifs = [23, 60, 131, 132, 133, 200, 224, 239, 327, 371]
for column in s3d_motif_df:
    if column.startswith('m'):
        column_motif = int(column[5:])
        if column_motif not in top_10_motifs:
            s3d_motif_df.drop(column, axis=1, inplace=True)

# Ignore the first column
s3d_motif_df_values = s3d_motif_df.iloc[:, 1:].values
# Normalize by row
row_sums = s3d_motif_df_values.sum(axis=1)
s3d_motif_df1 = s3d_motif_df_values / row_sums[:, np.newaxis]

# Combine with the first column
s3d_motif_df = pd.DataFrame(np.hstack((s3d_motif_df[['video']].values, s3d_motif_df1)), columns=s3d_motif_df.columns)

# ENS average between motif
s3d_ens_df = pd.read_csv(s3d_ens_path)
s3d_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# ENS per motif
s3d_ensm_df = pd.read_csv(s3d_ensm_path)
s3d_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0]) + 1}_m{x[6:]}' if 'motif' in x else x, inplace=True)
for column in s3d_ensm_df:
    if column.startswith('e'):
        column_motif = int(column[12:])
        if column_motif not in top_10_motifs:
            s3d_ensm_df.drop(column, axis=1, inplace=True)
# Entropy
# s3d_entropy_df = pd.read_csv(s3d_entropy_path)
# s3d_entropy_df.rename(columns=lambda x: f'entropy_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

s3d_df = pd.merge(bd_df, s3d_motif_df, on='video')
s3d_df = pd.merge(s3d_df, s3d_ens_df, on='video')
s3d_df = pd.merge(s3d_df, s3d_ensm_df, on='video')
# s3d_df = pd.merge(s3d_df, s3d_entropy_df, on='video')
s3d_df.drop('video', axis=1, inplace=True)

s3d_df['ens_diff'] = s3d_df['ens_epoch3'] - s3d_df['ens_epoch1']
# s3d_df['entropy_diff'] = s3d_df['entropy_epoch3'] - s3d_df['entropy_epoch1']

s3d_ensm_epoch3 = []
s3d_ensm_epoch1 = []
s3d_ensm_diff = []
for i in top_10_motifs:
    s3d_ensm_epoch3.append('ens_epoch3_m' + str(i))
    s3d_ensm_epoch1.append('ens_epoch3_m' + str(i))
    s3d_ensm_diff.append('ens_epoch3_m' + str(i))

# for i,v in enumerate(ensm_diff):
#     s3d_df[v] = s3d_df[ensm_epoch3[i]] - s3d_df[ensm_epoch1[i]]
for i in top_10_motifs:
    v = f'ens_epoch3_m{i}'
    s3d_df[v] = s3d_df[v] - s3d_df[v]
import scipy

s3d_motif_df_ = s3d_motif_df.copy()
drop_rows_if_50(s3d_motif_df_)
HC_motif_usage = s3d_motif_df_[bd_df['BD'] == 0]
HC_motif_usage.drop('video', axis=1, inplace=True)
BD_motif_usage = s3d_motif_df_[bd_df['BD'] == 1]
BD_motif_usage.drop('video', axis=1, inplace=True)
# %%
for motif_i_usage in HC_motif_usage.columns:
    s = stats.ttest_ind(HC_motif_usage[motif_i_usage], BD_motif_usage[motif_i_usage])
    print(f"{motif_i_usage}")
    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s.statistic, s.pvalue))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], YMRS_score[:n_subject_in_population])
    print(
        "          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
    print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score[0][0],
                                                                               corr_HAM_D_score[1]))
# %%
"""
MMACTION
"""
# %%
mmaction_motif_df = pd.read_csv(mmaction_motif_path)
mmaction_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)
top_10_motifs = [1, 6, 8, 11, 12, 14, 17, 48, 59, 80]
for column in mmaction_motif_df:
    if column.startswith('m'):
        column_motif = int(column[5:])
        if column_motif not in top_10_motifs:
            mmaction_motif_df.drop(column, axis=1, inplace=True)
# Ignore the first column
mmaction_motif_df_values = mmaction_motif_df.iloc[:, 1:].values
# Normalize by row
row_sums = mmaction_motif_df_values.sum(axis=1)
mmaction_motif_df1 = mmaction_motif_df_values / row_sums[:, np.newaxis]
# Combine with the first column
mmaction_motif_df = pd.DataFrame(np.hstack((mmaction_motif_df[['video']].values, mmaction_motif_df1)),
                                 columns=mmaction_motif_df.columns)

# ENS average between motif
mmaction_ens_df = pd.read_csv(mmaction_ens_path)
mmaction_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5]) + 1}' if 'split' in x else x, inplace=True)

# ENS per motif
mmaction_ensm_df = pd.read_csv(mmaction_ensm_path)
mmaction_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0]) + 1}_m{x[6:]}' if 'motif' in x else x, inplace=True)
for column in mmaction_ensm_df:
    if column.startswith('e'):
        column_motif = int(column[12:])
        if column_motif not in top_10_motifs:
            mmaction_ensm_df.drop(column, axis=1, inplace=True)

# mmaction_entropy_df = pd.read_csv(mmaction_entropy_path)
# mmaction_entropy_df.rename(columns=lambda x: f'entropy_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

mmaction_df = pd.merge(bd_df, mmaction_motif_df, on='video')
mmaction_df = pd.merge(mmaction_df, mmaction_ens_df, on='video')
mmaction_df = pd.merge(mmaction_df, mmaction_ensm_df, on='video')
# mmaction_df = pd.merge(mmaction_df, mmaction_entropy_df, on='video')
mmaction_df.drop('video', axis=1, inplace=True)

mmaction_df['ens_diff'] = mmaction_df['ens_epoch3'] - mmaction_df['ens_epoch1']
# mmaction_df['entropy_diff'] = mmaction_df['entropy_epoch3'] - mmaction_df['entropy_epoch1']

mmaction_ensm_epoch3 = []
mmaction_ensm_epoch1 = []
mmaction_ensm_diff = []
for i in top_10_motifs:
    mmaction_ensm_epoch3.append('ens_epoch3_m' + str(i))
    mmaction_ensm_epoch1.append('ens_epoch3_m' + str(i))
    mmaction_ensm_diff.append('ens_epoch3_m' + str(i))

# for i,v in enumerate(ensm_diff):
#     mmaction_df[v] = mmaction_df[ensm_epoch3[i]] - mmaction_df[ensm_epoch1[i]]
for i in top_10_motifs:
    v = f'ens_epoch3_m{i}'
    mmaction_df[v] = mmaction_df[v] - mmaction_df[v]

HC_motif_usage = mmaction_motif_df[bd_df['BD'] == 0]
HC_motif_usage.drop('video', axis=1, inplace=True)
BD_motif_usage = mmaction_motif_df[bd_df['BD'] == 1]
BD_motif_usage.drop('video', axis=1, inplace=True)
mmaction_motif_df_ = mmaction_motif_df.copy()
drop_rows_if_50(mmaction_motif_df_)
HC_motif_usage = mmaction_motif_df_[bd_df['BD'] == 0]
HC_motif_usage.drop('video', axis=1, inplace=True)
BD_motif_usage = mmaction_motif_df_[bd_df['BD'] == 1]
BD_motif_usage.drop('video', axis=1, inplace=True)
# %%
for motif_i_usage in HC_motif_usage.columns:
    s = stats.ttest_ind(HC_motif_usage[motif_i_usage], BD_motif_usage[motif_i_usage])
    print(f"{motif_i_usage}")
    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s.statistic, s.pvalue))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], YMRS_score[:n_subject_in_population])
    print(
        "          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
    print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_HAM_D_score[0][0],
                                                                               corr_HAM_D_score[1]))

# %% Quality check. Apply drop_rows_if_50 to all loaded CSVs

dlc_df = drop_rows_if_50(dlc_df)

hbpm_motif_df = drop_rows_if_50(hbpm_df)

s3d_df = drop_rows_if_50(s3d_df)

mmaction_df = drop_rows_if_50(mmaction_df)

spatialD_df = drop_rows_if_50(spatialD_df)
bd_df = drop_rows_if_50(bd_df)
# %%
assessmentNames = assessment_df.columns[2:]
# %% Compute the Entropy of dwell time of each approach
from scipy.stats import entropy

dfs = [vame_motif_df, mmaction_motif_df, s3d_motif_df, dlc_motif_df, hbpm_motif_df]
dfs_name = ['vame', 'mmaction', 's3d', 'dlc', 'hbpm']
entropy_values = []
entropy_values_std = []
entropy_values_all = []
entropy_values_all_std = []
entropy_of_BD_all = []
entropy_of_HC_all = []
for i, df in enumerate(dfs):
    df = pd.merge(bd_df, df, on='video')
    df = drop_rows_if_50(df)
    X = df.drop('video', axis=1)
    y = df['BD']
    BD_motif_usage = df[bd_df['BD'] == 1]
    BD_motif_usage.drop('video', axis=1, inplace=True)
    HC_motif_usage = df[bd_df['BD'] == 0]
    HC_motif_usage.drop('video', axis=1, inplace=True)
    BD_motif_usage_values = BD_motif_usage.values
    HC_motif_usage_values = HC_motif_usage.values

    entropy_of_BD = []
    for row in BD_motif_usage_values:
        probabilities = row.astype(float)
        entropy_of_BD.append(entropy(probabilities, base=2))  # Compute entropy
    entropy_of_BD_all.append(entropy_of_BD)

    entropy_of_HC = []
    for row in HC_motif_usage_values:
        probabilities = row.astype(float)
        entropy_of_HC.append(entropy(probabilities, base=2))  #
    entropy_of_HC_all.append(entropy_of_HC)

    entropy_values.append((np.nanmean(entropy_of_BD), np.nanmean(entropy_of_HC)))
    entropy_values_std.append((np.nanstd(entropy_of_BD), np.nanstd(entropy_of_HC)))

    entropy_of_all = []
    for row in [BD_motif_usage_values, HC_motif_usage_values]:
        probabilities = row.astype(float)
        entropy_of_all.append(entropy(probabilities, base=2))
    entropy_values_all.append(np.nanmean(entropy_of_all))
    entropy_values_all_std.append(np.nanstd(entropy_of_all))

from scipy.stats import ttest_ind

BD_p_values = []
for i in range(1, 5):  # Compare with sublists 1, 2, 3, 4
    _, p_val = ttest_ind(entropy_of_BD_all[0], entropy_of_BD_all[i], equal_var=False)
    BD_p_values.append(p_val)
    print(f"P-value between sublist 0 and sublist {i}: {p_val}")
HC_p_values = []
for i in range(1, 5):  # Compare with sublists 1, 2, 3, 4
    _, p_val = ttest_ind(entropy_of_HC_all[0], entropy_of_HC_all[i], equal_var=False)
    HC_p_values.append(p_val)
    print(f"P-value between sublist 0 and sublist {i}: {p_val}")

# Extracting mean and std_devs into separate lists
entropy_all = [x for x in entropy_values_all]
entropy_std_all = [x for x in entropy_values_all_std]

BD_mean_values = [x[0] for x in entropy_values]
BD_std_devs = [x[0] for x in entropy_values_std]
HC_mean_values = [x[1] for x in entropy_values]
HC_std_devs = [x[1] for x in entropy_values_std]
from pathlib import Path

# Plotting the bar graph
fig, ax = plt.subplots()
index = np.arange(len(dfs_name))
bar_width = 0.35
opacity = 0.8
b_o_colors = ['#1f77b4', '#ff7f0e']
# Plotting bars for mean
plt.errorbar(index, BD_mean_values, yerr=BD_std_devs, color=b_o_colors[1], label='BD', marker='o', linestyle='')
plt.errorbar(index + bar_width, HC_mean_values, yerr=HC_std_devs, color=b_o_colors[0], label='HC', marker='o',
             linestyle='')
plt.ylabel('Entropy')
plt.xticks(index + bar_width / 2, dfs_name)
plt.legend()
plt.tight_layout()
plt.show()
project_name = 'BD25-HC25-final-May17-2023'
n_cluster = 10
pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
Path(pwd).mkdir(parents=True, exist_ok=True)
fname = "dwell-time-entropy-{}.png".format(n_cluster)
fname_pdf = "dwell-time-entropy-{}.pdf".format(n_cluster)
fig.savefig(os.path.join(pwd, fname), transparent=True)
fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)

# plot total entropy
#
fig, ax = plt.subplots()
index = np.arange(len(dfs_name))

plt.errorbar(index, entropy_all, yerr=entropy_std_all, color='k', label='Entropy All', marker='o', linestyle='')
plt.ylabel('Entropy')
plt.xticks(index + bar_width / 2, dfs_name)
plt.legend()
plt.tight_layout()
plt.show()
pwd = r'{}\Behavior_VAE_data\{}\figure\dwell-time'.format(onedrive_path, project_name)
Path(pwd).mkdir(parents=True, exist_ok=True)
fname = "dwell-time-all-entropy-{}.png".format(n_cluster)
fname_pdf = "dwell-time-all-entropy-{}.pdf".format(n_cluster)
fig.savefig(os.path.join(pwd, fname), transparent=True)
fig.savefig(os.path.join(pwd, fname_pdf), transparent=True)
# %%
'''
Feature Selection with 4-CV (did not use after all)
'''

features = []
results = []
scaler = MinMaxScaler()
dfs = [vame_df, mmaction_df, s3d_df, dlc_df, hbpm_df, spatialD_df]
dfs_name = ['vame', 'mmaction', 's3d', 'dlc', 'hbpm', 'spatialD']
for i, df in enumerate(dfs):
    df[df.columns] = scaler.fit_transform(df)
    X = df.drop('BD', axis=1)
    y = df['BD']
    groups = bd_df['video']
    print(f"scaled data for {dfs_name[i]}")
    logreg = LogisticRegression()
    selector = SequentialFeatureSelector(logreg,
                                         n_features_to_select=30,
                                         direction="backward",
                                         scoring='accuracy',
                                         cv=4)  # cv=4
    print(f"selecting features for {dfs_name[i]}")
    selector.fit(X, y)
    selected_features = selector.get_support()
    top_features = list(X.columns[selected_features])
    print(f"{dfs_name[i]} selected {top_features}")
    features.append(top_features)
    print(f"Classify selected features in {dfs_name[i]}")
    results.append(classify(df, top_features, 100, random_seed)[0])

df = assessment_df
X = df.drop('BD', axis=1)
y = df['BD']
assessment_top_features = list(X.columns)
features.append(assessment_top_features)
print(f"Classify selected features in assessment ")
results.append(classify(df, assessment_top_features, 100, random_seed)[0])
# %%
'''
Feature Selection with LOSO-CV (did not use after all)
'''
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
import numpy as np

features = []
results = []
scaler = MinMaxScaler()

dfs = [vame_df, mmaction_df, s3d_df, dlc_df, hbpm_df]
dfs_name = ['vame', 'mmaction', 's3d', 'dlc', 'hbpm']

for i, df in enumerate(dfs):
    df[df.columns] = scaler.fit_transform(df)

    X = df.drop(columns=['BD'])
    y = df['BD']
    groups = bd_df['video']  # Use 'video' as the grouping factor for LOSO

    print(f"Selecting features for {dfs_name[i]} using standard 4-fold CV")

    # **Feature Selection with Standard 4-Fold CV**
    logreg = LogisticRegression()
    selector = SequentialFeatureSelector(logreg,
                                         n_features_to_select=30,
                                         direction="backward",
                                         scoring='accuracy',
                                         cv=4)  # Standard 4-fold CV for feature selection

    selector.fit(X, y)
    selected_features = selector.get_support()
    top_features = list(X.columns[selected_features])

    print(f"{dfs_name[i]} selected features: {top_features}")
    features.append(top_features)

    # **Classification and Validation using LOSO-CV**
    print(f"Classifying using LOSO-CV on {dfs_name[i]}")

    logo = LeaveOneGroupOut()
    logo_results = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx][top_features], X.iloc[test_idx][top_features]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        classifier = LogisticRegression(penalty='l2', C=0.1)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)  # Evaluate on test subject
        logo_results.append(score)

    mean_accuracy = np.mean(logo_results)  # Compute LOSO accuracy
    results.append(mean_accuracy)
    print(f"Mean LOSO-CV accuracy for {dfs_name[i]}: {mean_accuracy:.4f} +-{np.std(logo_results):.4f}")

# **Process the assessment dataset separately**
df = assessment_df
X = df.drop(columns=['BD'])
y = df['BD']
assessment_top_features = list(X.columns)
features.append(assessment_top_features)

print(f"Classifying assessment dataset using LOSO-CV")

logo_results = []
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx][assessment_top_features], X.iloc[test_idx][assessment_top_features]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    classifier = LogisticRegression(penalty='l2')
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    logo_results.append(score)

mean_accuracy = np.mean(logo_results)
results.append(mean_accuracy)
print(f"Mean LOSO-CV accuracy for assessment: {mean_accuracy:.4f}+-{np.std(logo_results):.4f}")

# **Process the spatialD dataset separately**
df = spatialD_df
X = df.drop(columns=['BD'])
y = df['BD']
spatialD_features = list(X.columns)
features.append(spatialD_features)
print(f"Classifying spatialD dataset using LOSO-CV")
logo = LeaveOneGroupOut()
for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx][spatialD_features], X.iloc[test_idx][spatialD_features]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    classifier = LogisticRegression(penalty='l2')
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    logo_results.append(score)

mean_accuracy = np.mean(logo_results)
results.append(mean_accuracy)
print(f"Mean LOSO-CV accuracy for spatialD: {mean_accuracy:.4f}+-{np.std(logo_results):.4f}")
# %%
'''
Feature Selection with LNSO-CV (we used this)
'''
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
import numpy as np

features = []
results = []
scaler = MinMaxScaler()

dfs = [vame_df, mmaction_df, s3d_df, dlc_df, hbpm_df]
dfs_name = ['vame', 'mmaction', 's3d', 'dlc', 'hbpm']

for i, df in enumerate(dfs):
    df[df.columns] = scaler.fit_transform(df)

    X = df.drop(columns=['BD'])
    y = df['BD']
    groups = bd_df['video']  # Using 'video' as subject ID

    print(f"Selecting features for {dfs_name[i]} using standard 4-fold CV")

    # **Step 1: Feature Selection (Using 4-Fold CV)**
    logreg = LogisticRegression()
    selector = SequentialFeatureSelector(logreg,
                                         n_features_to_select=25,
                                         direction="backward",
                                         scoring='accuracy',
                                         cv=4)  # Standard 4-fold CV for feature selection

    selector.fit(X, y)
    selected_features = selector.get_support()
    top_features = list(X.columns[selected_features])

    print(f"{dfs_name[i]} selected features: {top_features}")
    features.append(top_features)

    # **Step 2: Classification Using Leave-5-Subjects-Out (L5SO-CV)**
    print(f"Classifying using L5SO-CV on {dfs_name[i]}")

    l5so = GroupKFold(n_splits=len(np.unique(groups)) // 5)  # Splitting into groups of 5 subjects
    l5so_results = []

    for train_idx, test_idx in l5so.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx][top_features], X.iloc[test_idx][top_features]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        classifier = LogisticRegression(penalty='l2')
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)  # Evaluate on test subjects
        l5so_results.append(score)

    mean_accuracy = np.mean(l5so_results)  # Compute Leave-5-Subjects-Out accuracy
    results.append(mean_accuracy)
    print(f"Mean L5SO-CV accuracy for {dfs_name[i]}: {mean_accuracy:.4f} +- {np.std(l5so_results):.4f}")

# **Process the assessment dataset separately**
df = assessment_df
X = df.drop(columns=['BD'])
y = df['BD']
assessment_top_features = list(X.columns)
features.append(assessment_top_features)

print(f"Classifying assessment dataset using L5SO-CV")
l5so_results = []

for train_idx, test_idx in l5so.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx][assessment_top_features], X.iloc[test_idx][assessment_top_features]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    classifier = LogisticRegression(penalty='l2')
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    l5so_results.append(score)

mean_accuracy = np.mean(l5so_results)
results.append(mean_accuracy)
print(f"Mean L5SO-CV accuracy for assessment: {mean_accuracy:.4f} +- {np.std(l5so_results):.4f}")

# **Process the spatialD dataset separately**
df = spatialD_df
X = df.drop(columns=['BD'])
y = df['BD']
spatialD_features = list(X.columns)
features.append(spatialD_features)
print(f"Classifying spatialD dataset using LOSO-CV")
l5so_results = []
for train_idx, test_idx in l5so.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx][spatialD_features], X.iloc[test_idx][spatialD_features]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    classifier = LogisticRegression(penalty='l2')
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    l5so_results.append(score)

mean_accuracy = np.mean(l5so_results)
results.append(mean_accuracy)
print(f"Mean LOSO-CV accuracy for spatialD: {mean_accuracy:.4f}+-{np.std(l5so_results):.4f}")
# %%
score_path = rf"{data_path}\all_approaches_30feature_selected_scores_seed_{random_seed}_rm_entropy"
features_path = rf"{data_path}\all_approaches_30features_seed_{random_seed}_rm_entropy"
np.save(score_path, np.array(results))
np.save(features_path, np.array(features, dtype=object))

# %%

'''
Significant Tests
'''

score_path = rf"{data_path}\all_approaches_15feature_selected_scores_seed_{random_seed}_rm_entropy.npy"
results = np.load(score_path)
samples = list(itertools.chain.from_iterable(results))
labels = ['vame'] * 400 + ['mmaction'] * 400 + ['s3d'] * 400 + ['dlc'] * 400 + ['hbpm'] * 400 + ['assessment'] * 400

result = pairwise_tukeyhsd(samples, labels)

print(result)
print(result.pvalues[4])
print(result.pvalues[8])
print(result.pvalues[11])
print(result.pvalues[13])
print(result.pvalues[14])

# #%%
# res = tukey_hsd(results[0], results[1], results[2], results[3], results[4], results[5])
#
# #%%
# print(res)
# #%%
# print(res.pvalue)
#
# #%%
# pvalue_indices = [4, 8, 11, 13, 14]
#
# for idx in pvalue_indices:
#     # Get the p-value at the specified index and format it
#     pvalue = result.pvalues[idx]
#     formatted_pvalue = f"{pvalue:.500f}"  # Adjust the number of decimal places as needed
#     print(f"Formatted p-value at index {idx}: {formatted_pvalue}")
#
