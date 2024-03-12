#%%
import csv
import numpy as np
import pandas as pd
import os
import seaborn as sns 
import matplotlib.pyplot as plt
import random
import copy
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_validate, cross_val_predict

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold

import statsmodels.api as sm

#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]


#%% v - vame, d - DLC, h - hBPM, s - S3D, m - MMAction
diagnostic_path = r"C:\Users\kietc\SURF\jack-data\scaled_diagnostic_data.csv"

vame_motif_path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"
vame_ens_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\ens_3_split.csv'
vame_ensm_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\ens_per_motif_3_split.csv'
vame_entropy_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
vame_count_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\count_3_split.csv'
vame_volume_path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\volume.csv"

dlc_motif_path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\motif_usage_overall.csv"
dlc_entropy_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\entropy_3_split.csv'
dlc_ens_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\ens_3_split.csv'

hbpm_motif_path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
hbpm_entropy_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\entropy_3_split.csv'
hbpm_ens_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\ens_3_split.csv'
hbpm_count_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\count_3_split.csv'

s3d_motif_path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
s3d_entropy_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\entropy_3_split.csv'
s3d_ens_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\ens_3_split.csv'
s3d_count_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\count_3_split.csv'

mmaction_motif_path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"
mmaction_entropy_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\entropy_3_split.csv'
mmaction_ens_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\ens_3_split.csv'
mmaction_count_path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\count_3_split.csv'


export_path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification\{}"
export_result_path =  r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification\{}"

# %%
def classify(df, features):
    X = df[features]
    y = df['BD']

    model = LogisticRegression(max_iter=200)
    y_pred_cv = cross_val_predict(model, X, y, cv=4)
    classification_rep_cv = classification_report(y, y_pred_cv, digits=4)

    print(f'Classification Report (Cross-Validation):\n{classification_rep_cv}') 

    return [classification_rep_cv['accuracy'], classification_rep_cv['weighted avg']['precision'], classification_rep_cv['weighted avg']['recall']]


# %%
"""
PANDA 🐼
"""

#%% 
assessment_df = pd.read_csv(diagnostic_path)
assessment_df.drop('gender', axis=1, inplace=True)
bd_df = assessment_df[['video', 'BD']]

"""
VAME
"""
#%%
# Motif
vame_motif_df = pd.read_csv(vame_motif_path)
vame_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS average between motif
vame_ens_df = pd.read_csv(vame_ens_path)
vame_ens_df.rename(columns=lambda x: f'ens_s{x[5]}' if 'split' in x else x, inplace=True)

# ENS per motif
vame_ensm_df = pd.read_csv(vame_ensm_path)
vame_ensm_df.rename(columns=lambda x: f'ens_s{x[0]}_m{x[6:]}' if 'motif' in x else x, inplace=True)

# Entropy
vame_entropy_df = pd.read_csv(vame_entropy_path)
vame_entropy_df.rename(columns=lambda x: f'ent_s{x[5]}' if 'split' in x else x, inplace=True)

# Count
vame_count_df = pd.read_csv(vame_count_path)
vame_count_df.rename(columns=lambda x: f'cnt_s{x[5]}' if 'split' in x else x, inplace=True)

#Volume
vame_volume_df = pd.read_csv(vame_volume_path)
vame_count_df.rename(columns=lambda x: f'v_s{x[5]}' if 'split' in x else x, inplace=True)

vame_df = pd.merge(bd_df, vame_motif_df, on='video')
vame_df = pd.merge(vame_df, vame_ens_df, on='video')
vame_df = pd.merge(vame_df, vame_ensm_df, on='video')
vame_df = pd.merge(vame_df, vame_entropy_df, on='video')
vame_df = pd.merge(vame_df, vame_count_df, on='video')
vame_df = pd.merge(vame_df, vame_volume_df, on='video')
vame_df.drop('video', axis=1, inplace=True)

#%%
vame_df['ens_diff'] = vame_df['ens_s2'] - vame_df['ens_s0'] 
vame_df['ent_diff'] = vame_df['ent_s2'] - vame_df['ent_s0']

#%%
ensm_s2 = ['ens_s2_m0', 'ens_s2_m1', 'ens_s2_m2', 'ens_s2_m3', 'ens_s2_m4',
        'ens_s2_m5', 'ens_s2_m6', 'ens_s2_m7', 'ens_s2_m8', 'ens_s2_m9']
ensm_s0 = ['ens_s0_m0', 'ens_s0_m1', 'ens_s0_m2', 'ens_s0_m3', 'ens_s0_m4',
        'ens_s0_m5', 'ens_s0_m6', 'ens_s0_m7', 'ens_s0_m8', 'ens_s0_m9']
ensm_diff = ['ens_diff_m0', 'ens_diff_m1', 'ens_diff_m2', 'ens_diff_m3', 'ens_diff_m4',
        'ens_diff_m5', 'ens_diff_m6', 'ens_diff_m7', 'ens_diff_m8', 'ens_diff_m9']

for i,v in enumerate(ensm_diff):
    vame_df[v] = vame_df[ensm_s2[i]] - vame_df[ensm_s0[i]]


#%%
"""
DLC
"""
#%%
dlc_motif_df = pd.read_csv(dlc_motif_path)
dlc_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS average between motif
dlc_ens_df = pd.read_csv(dlc_ens_path)
dlc_ens_df.rename(columns=lambda x: f'ens_s{x[5]}' if 'split' in x else x, inplace=True)
# Entropy
dlc_entropy_df = pd.read_csv(dlc_entropy_path)
dlc_entropy_df.rename(columns=lambda x: f'ent_s{x[5]}' if 'split' in x else x, inplace=True)

dlc_df = pd.merge(bd_df, dlc_motif_df, on='video')
dlc_df = pd.merge(dlc_df, dlc_ens_df, on='video')
dlc_df = pd.merge(dlc_df, dlc_entropy_df, on='video')
dlc_df.drop('video', axis=1, inplace=True)

#%%
dlc_df['ens_diff'] = dlc_df['ens_s2'] - dlc_df['ens_s0'] 
dlc_df['ent_diff'] = dlc_df['ent_s2'] - dlc_df['ent_s0']

#%%
"""
HBPM
"""
#%%
hbpm_motif_df = pd.read_csv(hbpm_motif_path)
hbpm_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS average between motif
hbpm_ens_df = pd.read_csv(hbpm_ens_path)
hbpm_ens_df.rename(columns=lambda x: f'ens_s{x[5]}' if 'split' in x else x, inplace=True)

# Entropy
hbpm_entropy_df = pd.read_csv(hbpm_entropy_path)
hbpm_entropy_df.rename(columns=lambda x: f'ent_s{x[5]}' if 'split' in x else x, inplace=True)

hbpm_df = pd.merge(bd_df, hbpm_motif_df, on='video')
hbpm_df = pd.merge(hbpm_df, hbpm_ens_df, on='video')
hbpm_df = pd.merge(hbpm_df, hbpm_entropy_df, on='video')
hbpm_df.drop('video', axis=1, inplace=True)

#%%
hbpm_df['ens_diff'] = hbpm_df['ens_s2'] - hbpm_df['ens_s0'] 
hbpm_df['ent_diff'] = hbpm_df['ent_s2'] - hbpm_df['ent_s0']


#%%
"""
S3D
"""

#%%
s3d_motif_df = pd.read_csv(s3d_motif_path)
s3d_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS average between motif
s3d_ens_df = pd.read_csv(s3d_ens_path)
s3d_ens_df.rename(columns=lambda x: f'ens_s{x[5]}' if 'split' in x else x, inplace=True)

# Entropy
s3d_entropy_df = pd.read_csv(s3d_entropy_path)
s3d_entropy_df.rename(columns=lambda x: f'ent_s{x[5]}' if 'split' in x else x, inplace=True)

s3d_df = pd.merge(bd_df, s3d_motif_df, on='video')
s3d_df = pd.merge(s3d_df, s3d_ens_df, on='video')
s3d_df = pd.merge(s3d_df, s3d_entropy_df, on='video')
s3d_df.drop('video', axis=1, inplace=True)

#%%
s3d_df['ens_diff'] = s3d_df['ens_s2'] - s3d_df['ens_s0'] 
s3d_df['ent_diff'] = s3d_df['ent_s2'] - s3d_df['ent_s0']


#%%
"""
MMACTION
"""
#%%
mmaction_motif_df = pd.read_csv(mmaction_motif_path)
mmaction_motif_df.rename(columns=lambda x: f'motif{x[2:]}' if x.startswith('0m') else x, inplace=True)

# ENS average between motif
mmaction_ens_df = pd.read_csv(mmaction_ens_path)
mmaction_ens_df.rename(columns=lambda x: f'ens_s{x[5]}' if 'split' in x else x, inplace=True)

mmaction_entropy_df = pd.read_csv(mmaction_entropy_path)
mmaction_entropy_df.rename(columns=lambda x: f'ent_s{x[5]}' if 'split' in x else x, inplace=True)

mmaction_df = pd.merge(bd_df, mmaction_motif_df, on='video')
mmaction_df = pd.merge(mmaction_df, mmaction_ens_df, on='video')
mmaction_df = pd.merge(mmaction_df, mmaction_entropy_df, on='video')
mmaction_df.drop('video', axis=1, inplace=True)

#%%
mmaction_df['ens_diff'] = mmaction_df['ens_s2'] - mmaction_df['ens_s0'] 
mmaction_df['ent_diff'] = mmaction_df['ent_s2'] - mmaction_df['ent_s0']


#%% OLS
## use this approach for assessment scores?
X = vame_df.drop('BD', axis=1)
y = vame_df['BD']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# %% Lasso Regression
# https://medium.com/@agrawalsam1997/feature-selection-using-lasso-regression-10f49c973f08
X = vame_df.drop('BD', axis=1).values
y = vame_df['BD'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.24, random_state=42, stratify=y)

# %%
# parameters to be tested on GridSearchCV
params = {"alpha":np.arange(0.00001, 10, 500)}

# Number of Folds and adding the random state for replication
kf=KFold(n_splits=5,shuffle=True, random_state=42)

# Initializing the Model
lasso = Lasso()

# GridSearchCV with model, params and folds.
lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X, y)
print("Best Params {}".format(lasso_cv.best_params_))

names=vame_df.drop("BD", axis=1).columns
print("Column Names: {}".format(names.values))

#%%
# calling the model with the best parameter
lasso1 = Lasso(alpha=0.00001)
lasso1.fit(X_train, y_train)

# Using np.abs() to make coefficients positive.  
lasso1_coef = np.abs(lasso1.coef_)

# plotting the Column Names and Importance of Columns. 
plt.bar(names, lasso1_coef)
plt.xticks(rotation=90)
plt.grid()
plt.title("Feature Selection Based on Lasso")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()


# %% Correlation coef
correlation_matrix = vame_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# %%
motif_columns = vame_motif_df.columns[1:]
classify(vame_df, motif_columns)
# %%
