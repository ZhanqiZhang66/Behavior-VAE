#%%
import csv
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import random
import copy
import json
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold

import statsmodels.api as sm
# from scipy.stats import f_oneway, tukey_hsd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from data.load_data import load_pt_data
import itertools
from sklearn.preprocessing import scale
#%% Path
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
#%% Data Path
random.seed(42)
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]


random_seed = 14
diagnostic_path = rf"{data_path}\scaled_diagnostic_data.csv"
assessment_score_path = rf"{data_path}\assessment_scores_seed_{random_seed}"

vame_motif_path = rf"{data_path}\VAME\motif_usage_overall.csv"
vame_ens_path = rf'{data_path}\VAME\ens_3_split.csv'
# ens per motif
vame_ensm_path = rf'{data_path}\VAME\ens_per_motif_3_split.csv'
vame_entropy_path = rf'{data_path}\VAME\entropy_3_split.csv'
vame_count_path = rf'{data_path}\VAME\count_3_split.csv'
vame_volume_path = rf"{data_path}\VAME\volume.csv"
vame_volume_per_motif_path = rf'{data_path}\VAME\volume_per_motif.csv'
vame_feature_selection_path = rf"{data_path}\VAME\feature_selection.csv"
vame_score_path = rf"{data_path}\VAME\scores_seed_{random_seed}"

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
export_result_path =  rf"{data_path}\Classification\{random_seed}"

ensm_epoch3 = ['ens_epoch3_m0', 'ens_epoch3_m1', 'ens_epoch3_m2', 'ens_epoch3_m3', 'ens_epoch3_m4',
        'ens_epoch3_m5', 'ens_epoch3_m6', 'ens_epoch3_m7', 'ens_epoch3_m8', 'ens_epoch3_m9']
ensm_epoch1 = ['ens_epoch1_m0', 'ens_epoch1_m1', 'ens_epoch1_m2', 'ens_epoch1_m3', 'ens_epoch1_m4',
        'ens_epoch1_m5', 'ens_epoch1_m6', 'ens_epoch1_m7', 'ens_epoch1_m8', 'ens_epoch1_m9']
ensm_diff = ['ens_diff_m0', 'ens_diff_m1', 'ens_diff_m2', 'ens_diff_m3', 'ens_diff_m4',
        'ens_diff_m5', 'ens_diff_m6', 'ens_diff_m7', 'ens_diff_m8', 'ens_diff_m9']
n_subject_in_population = 25
data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
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
        scores = cross_validate(classifier, xtrain, ytrain, scoring=scoring, cv = 3)

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
                                    tol=tol, # a hyperparameter that gives < half of all features
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

#%% Reading assessment data

assessment_df = pd.read_csv(diagnostic_path)
assessment_df.drop('gender', axis=1, inplace=True)

bd_df = assessment_df[['video', 'BD']]
assessment_df.drop('video', axis=1, inplace=True)

#%% Reading VAME data


#%%
# Motif dwell time
vame_motif_df = pd.read_csv(vame_motif_path)
vame_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS(transition matrix)
vame_ens_df = pd.read_csv(vame_ens_path)
vame_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

# ENS(each row of motif in transition matrix)
vame_ensm_df = pd.read_csv(vame_ensm_path)
vame_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0])+1}_m{x[6:]}' if 'motif' in x else x, inplace=True)

# Entropy of transition matrix
vame_entropy_df = pd.read_csv(vame_entropy_path)
vame_entropy_df.rename(columns=lambda x: f'entropy_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

# Count of zeros in transition matrix
vame_count_df = pd.read_csv(vame_count_path)
vame_count_df.rename(columns=lambda x: f'cnt_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

# Volume of each latent vector
vame_volume_df = pd.read_csv(vame_volume_path)
vame_volume_df.rename(columns=lambda x: f'vol_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

#%%
vame_volume_per_motif_df = pd.read_csv(vame_volume_per_motif_path)

#%%
vame_volume_per_motif_df.rename(columns=lambda x: f'vol_epoch{int(x[0])+1}_motif{(x[6:])}' if 'motif' in x else x, inplace=True)

#%%
vame_df = pd.merge(bd_df, vame_motif_df, on='video')
vame_df = pd.merge(vame_df, vame_ens_df, on='video')
vame_df = pd.merge(vame_df, vame_ensm_df, on='video')
# vame_df = pd.merge(vame_df, vame_entropy_df, on='video')

#%%
vame_df = pd.merge(vame_df, vame_count_df, on='video')
vame_df = pd.merge(vame_df, vame_volume_df, on='video')
#%%
vame_df = pd.merge(vame_df, vame_volume_per_motif_df, on='video')
#%%
vame_df.drop('video', axis=1, inplace=True)

vame_df['ens_diff'] = vame_df['ens_epoch3'] - vame_df['ens_epoch1']
# vame_df['entropy_diff'] = vame_df['entropy_epoch3'] - vame_df['entropy_epoch1']
vame_df['vol_diff_2_1'] = vame_df['vol_epoch3'] - vame_df['vol_epoch2']
vame_df['vol_diff_1_0'] = vame_df['vol_epoch2'] - vame_df['vol_epoch1']
vame_df['vol_diff_2_0'] = vame_df['vol_epoch3'] - vame_df['vol_epoch1']

for i,v in enumerate(ensm_diff):
    vame_df[v] = vame_df[ensm_epoch3[i]] - vame_df[ensm_epoch1[i]]

# for i,v in enumerate(volume_per_motif_diff):
#     vame_df[v] = vame_df[volume_per_motif_epoch3[i]] - vame_df[volume_per_motif_epoch1[i]]

#%%
"""
DLC
"""
#%%
dlc_motif_df = pd.read_csv(dlc_motif_path)
dlc_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS average between motif
dlc_ens_df = pd.read_csv(dlc_ens_path)
dlc_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

# ENS per motif
dlc_ensm_df = pd.read_csv(dlc_ensm_path)
dlc_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0])+1}_m{x[6:]}' if 'motif' in x else x, inplace=True)

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

for i,v in enumerate(ensm_diff):
    dlc_df[v] = dlc_df[ensm_epoch3[i]] - dlc_df[ensm_epoch1[i]]


#%%
"""
HBPM
"""
#%%
hbpm_motif_df = pd.read_csv(hbpm_motif_path)
hbpm_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)
# Ignore the first column
hbpm_motif_df_values = hbpm_motif_df.iloc[:, 1:].values
# Normalize by row
row_sums = hbpm_motif_df_values.sum(axis=1)
hbpm_motif_df1= hbpm_motif_df_values / row_sums[:, np.newaxis]
# Combine with the first column
hbpm_motif_df = pd.DataFrame(np.hstack((hbpm_motif_df[['video']].values, hbpm_motif_df1)), columns=hbpm_motif_df.columns)


# ENS average between motif
hbpm_ens_df = pd.read_csv(hbpm_ens_path)
hbpm_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

# ENS per motif
hbpm_ensm_df = pd.read_csv(hbpm_ensm_path)
hbpm_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0])+1}_m{x[6:]}' if 'motif' in x else x, inplace=True)

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

for i,v in enumerate(ensm_diff):
    hbpm_df[v] = hbpm_df[ensm_epoch3[i]] - hbpm_df[ensm_epoch1[i]]

#%%
"""
S3D
"""
#%%
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
s3d_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)


# ENS per motif
s3d_ensm_df = pd.read_csv(s3d_ensm_path)
s3d_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0])+1}_m{x[6:]}' if 'motif' in x else x, inplace=True)
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
HC_motif_usage = s3d_motif_df[bd_df['BD']==0]
HC_motif_usage.drop('video', axis=1, inplace=True)
BD_motif_usage = s3d_motif_df[bd_df['BD']==1]
BD_motif_usage.drop('video', axis=1, inplace=True)
#%%
for motif_i_usage in HC_motif_usage.columns:
    s = stats.ttest_ind(HC_motif_usage[motif_i_usage], BD_motif_usage[motif_i_usage])
    print(f"{motif_i_usage}")
    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s.statistic, s.pvalue))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], YMRS_score[:n_subject_in_population])
    print("          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
    print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score[0][0], corr_HAM_D_score[1]))
#%%
"""
MMACTION
"""
#%%
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
mmaction_motif_df = pd.DataFrame(np.hstack((mmaction_motif_df[['video']].values, mmaction_motif_df1)), columns=mmaction_motif_df.columns)

# ENS average between motif
mmaction_ens_df = pd.read_csv(mmaction_ens_path)
mmaction_ens_df.rename(columns=lambda x: f'ens_epoch{int(x[5])+1}' if 'split' in x else x, inplace=True)

# ENS per motif
mmaction_ensm_df = pd.read_csv(mmaction_ensm_path)
mmaction_ensm_df.rename(columns=lambda x: f'ens_epoch{int(x[0])+1}_m{x[6:]}' if 'motif' in x else x, inplace=True)
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

HC_motif_usage = mmaction_motif_df[bd_df['BD']==0]
HC_motif_usage.drop('video', axis=1, inplace=True)
BD_motif_usage = mmaction_motif_df[bd_df['BD']==1]
BD_motif_usage.drop('video', axis=1, inplace=True)
#%%
for motif_i_usage in HC_motif_usage.columns:
    s = stats.ttest_ind(HC_motif_usage[motif_i_usage], BD_motif_usage[motif_i_usage])
    print(f"{motif_i_usage}")
    print("2 sample t-stat: {:.2f}, p-val: {:.3f}".format(s.statistic, s.pvalue))
    # print("motif  {}, permutation_test: {:.2f}, p-val: {:.3f}".format(i,res.statistic, res.pvalue))
    corr_HAM_D_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], HAM_D_score[:n_subject_in_population])
    corr_YMRS_score = scipy.stats.pearsonr(BD_motif_usage[motif_i_usage], YMRS_score[:n_subject_in_population])
    print("          Pearson corr YMARS-HC: rho: {:.2f}, p-val: {:.2f}".format(corr_YMRS_score[0][0], corr_YMRS_score[1]))
    print("          Pearson corr HAM_D-HC: rho: {:.2f}, p-val: {:.2f}".format (corr_HAM_D_score[0][0], corr_HAM_D_score[1]))
#%%
assessmentNames = assessment_df.columns[2:]
#%% Compute the Entropy of dwell time of each approach
from scipy.stats import entropy
dfs = [vame_motif_df, mmaction_motif_df, s3d_motif_df, dlc_motif_df, hbpm_motif_df]
dfs_name =  ['vame', 'mmaction','s3d', 'dlc', 'hbpm']
entropy_values = []
entropy_values_std = []
for i, df in enumerate(dfs):
    df = pd.merge(bd_df, df, on='video')
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
    entropy_of_HC = []
    for row in HC_motif_usage_values:
        probabilities = row.astype(float)
        entropy_of_HC.append(entropy(probabilities, base=2))  #
    entropy_values.append((np.mean(entropy_of_BD), np.mean(entropy_of_HC)))
    entropy_values_std.append((np.std(entropy_of_BD), np.std(entropy_of_HC)))

# Extracting mean and std_devs into separate lists
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
plt.errorbar(index + bar_width, HC_mean_values, yerr=HC_std_devs, color=b_o_colors[0], label='HC', marker='o', linestyle='')
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
#%%
'''
Feature Selection
'''

features = []
results = []
scaler = MinMaxScaler()
dfs = [vame_df, mmaction_df, s3d_df, dlc_df, hbpm_df]
dfs_name =  ['vame', 'mmaction','s3d', 'dlc', 'hbpm']
for i, df in enumerate(dfs):
    df[df.columns] = scaler.fit_transform(df)
    X = df.drop('BD', axis=1)
    y = df['BD']
    print(f"scaled data for {dfs_name[i]}")
    logreg = LogisticRegression()
    selector = SequentialFeatureSelector(logreg,
                                         n_features_to_select=15,
                                         direction="backward",
                                         scoring='accuracy',
                                         cv=4)
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

#%%
score_path = rf"{data_path}\all_approaches_15feature_selected_scores_seed_{random_seed}_rm_entropy"
features_path = rf"{data_path}\all_approaches_15features_seed_{random_seed}_rm_entropy"
np.save(score_path, np.array(results))
np.save(features_path, np.array(features, dtype=object))

#%%

'''
Significant Tests
'''

score_path = rf"{data_path}\all_approaches_15feature_selected_scores_seed_{random_seed}_rm_entropy.npy"
results = np.load(score_path)
samples = list(itertools.chain.from_iterable(results))
labels = ['vame'] * 400 + ['mmaction'] * 400 +  ['s3d'] * 400 + ['dlc'] * 400 + ['hbpm'] * 400 +  ['assessment'] * 400

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
