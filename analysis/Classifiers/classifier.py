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
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate

#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

#%%
def generate_volume(path):
    volume = {}
    for v in videos:
        latent = np.load(path.format(v, v))
        volume[v] = np.trace(np.cov(latent.T))
    return volume

def combine_df(df1, df2):
    copyDf1 = copy.deepcopy(df1)
    for key in copyDf1:
        copyDf1[key].extend(df2[key])
    return copyDf1

def load_data(path,  idf = None, end = 0, scale = 1):
    df = {}
    if idf:
        df = copy.deepcopy(idf)
    with open(path, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        next(csvreader)
        for row in csvreader:
            if end:
                vec = np.array(row[1:end]).astype(float).tolist()
            else:
                vec = np.array(row[1:]).astype(float).tolist()
            vec = [x/scale for x in vec]
            if row[0] not in df:
                df[row[0]] = []
            df[row[0]].extend(vec)
    return df

def epoch_sub(idf, indexA, indexB, size = 1):
    df = {}
    for v in videos:
        df[v] = []
        for i in range(size):
            df[v].append(idf[v][indexB + i] - idf[v][indexA +  i])
    return df

def classify(dataframe):
    X = []
    Y = []
    seeds = []
    for video in dataframe:
        X.append(dataframe[video][1:])
        Y.append(dataframe[video][0])

    acc = []
    pre = []
    rec = []
    score = []

    rand = random.randrange(100)
    for i in range(rand, rand + 50):
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.24, stratify=Y, random_state=i)

        # x
        sc_x = StandardScaler()
        xtrain = sc_x.fit_transform(xtrain)
        xtest = sc_x.transform(xtest)

        # model
        classifier = LogisticRegression(random_state = 0)

        
        classifier.fit(xtrain, ytrain)
        y_pred = classifier.predict(xtest)
    
        acc.append(accuracy_score(ytest, y_pred))
        pre.append(precision_score(ytest, y_pred))
        rec.append(recall_score(ytest, y_pred))

        # cross validation
        scoring = ['accuracy', 'precision', 'recall']
        scores = cross_validate(classifier, xtrain, ytrain, scoring = scoring, cv = 3)
        
        acc.extend(scores['test_accuracy'])
        pre.extend(scores['test_precision'])
        rec.extend(scores['test_recall'])
        seeds.append(i)
        score.append(classifier.score(xtest, ytest))

    print('Accuracy: %.05f (%.05f)' % (np.mean(acc), np.std(acc)))
    print('Precision: %.05f (%.05f)' % (np.mean(pre), np.std(pre)))
    print('Recall: %.05f (%.05f)' % (np.mean(rec), np.std(rec)))
    data = [acc, pre, rec, score]
    return data

def save_classifier_results(path, data):
    print(path)
    np.save(path, np.array(data))

def load_result(path):
    return np.load(path).tolist()

def graph_classifier_results(path, data, title):
    df_list = []
    for data_name, data_set in data.items():
        for j, metric in enumerate(['acc', 'pre', 'rec']):
            df_list.extend({
                'Data': data_name,
                'Metric': metric,
                'Score': score
            } for score in data_set[j])
    df = pd.DataFrame(df_list)
    # Create a boxplot using Seaborn
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.boxplot(x='Data', y='Score', hue='Metric', data=df, palette='Set3')

    # Add labels and title
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title(title)

    # Show the plot
    if path:
        plt.savefig(path + '.png', transparent=True)
        plt.savefig(path + '.pdf', transparent=True)
    
    plt.show()

#%% v - vame, d - DLC, h - hBPM, s - S3D, m - MMAction
diagnosticPath = r"C:\Users\kietc\SURF\jack-data\scaled_diagnostic_data.csv"

vMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"
dMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\motif_usage_overall.csv"
hMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
sMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
mMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"

vEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
dEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\entropy_3_split.csv'
hEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\entropy_3_split.csv'
sEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\entropy_3_split.csv'
mEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\entropy_3_split.csv'

vENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\ens_3_split.csv'
dENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\ens_3_split.csv'
hENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\ens_3_split.csv'
sENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\ens_3_split.csv'
mENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\ens_3_split.csv'

vCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\count_3_split.csv'
vVolumePath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\volume.csv"

exportPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification\{}"
exportResultPath =  r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification\{}"

#%%
hCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\count_3_split.csv'
sCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\count_3_split.csv'
mCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\count_3_split.csv'


#%% Create DF for each model
BD = load_data(diagnosticPath, end=2)
assessment = load_data(diagnosticPath, end=-1)

#%% Models - Motif Usage
VAME_Motif = load_data(vMotifPath, idf=BD, scale=27000)
DLC_Motif = load_data(dMotifPath, idf=BD, scale=27000)
HBPM_Motif = load_data(hMotifPath, idf=BD, scale=27000)
S3D_Motif = load_data(sMotifPath, idf=BD, scale=27000)
MMAction_Motif = load_data(mMotifPath, idf=BD, scale=27000)

#%% Models - ENS
VAME_ENS = load_data(vENSPath, idf=BD)
DLC_ENS = load_data(dENSPath, idf=BD)
HBPM_ENS = load_data(hENSPath, idf=BD)
S3D_ENS = load_data(sENSPath, idf=BD)
MMAction_ENS = load_data(mENSPath, idf=BD)

#%% Models - ENS epoch 3 - epoch 1
VAME_ENS1 = epoch_sub(VAME_ENS, 3, 1)
VAME_ENS1 = combine_df(BD, VAME_ENS1)

DLC_ENS1 = epoch_sub(DLC_ENS, 3, 1)
DLC_ENS1 = combine_df(BD, DLC_ENS1)

HBPM_ENS1 = epoch_sub(HBPM_ENS, 3, 1)
HBPM_ENS1  = combine_df(BD, HBPM_ENS1)

S3D_ENS1 = epoch_sub(S3D_ENS, 3, 1)
S3D_ENS1 = combine_df(BD, S3D_ENS1)

MMAction_ENS1 = epoch_sub(MMAction_ENS, 3, 1)
MMAction_ENS1 = combine_df(BD, MMAction_ENS1)

#%% VAME
VAME_Entropy = load_data(vEntropyPath, idf=BD)
VAME_Volume = load_data(vVolumePath, idf=BD)
VAME_Count = load_data(vCountPath, idf=BD)


#%% Assessment + VAME
VAME_AM = load_data(vMotifPath, idf=assessment, scale=27000)
VAME_AENS = load_data(vENSPath, idf=assessment)
VAME_AE = load_data(vEntropyPath, idf=assessment)
VAME_AV = load_data(vVolumePath, idf=assessment)
VAME_AC = load_data(vCountPath, idf=assessment)

#%% Assessment + VAME+
VAME_AME = load_data(vENSPath, idf=VAME_AM)
VAME_AMEV = load_data(vVolumePath, idf=VAME_AME)
VAME_AMEVC = load_data(vCountPath, idf=VAME_AMEV)



#%% Assessment
print("Assessment")
c_assessment = classify(assessment)
print("=======================")
save_classifier_results(exportPath.format("assessments_scales_50.npy"), c_assessment)

#%%
c_assessment = 

#%% Motif
print("VAME Motif")
c_VAME_Motif = classify(VAME_Motif)
print("DLC Motif")
c_DLC_Motif = classify(DLC_Motif)
print("hBPM Motif")
c_HBPM_Motif = classify(HBPM_Motif)
print("S3D Motif")
c_S3D_Motif = classify(S3D_Motif)
print("MMAction Motif")
c_MMAction_Motif = classify(MMAction_Motif)
print("=======================")
save_classifier_results(exportPath.format("vame_motif_50.npy"), c_VAME_Motif)
save_classifier_results(exportPath.format("dlc_motif_50.npy"), c_DLC_Motif)
save_classifier_results(exportPath.format("hbpm_motif_50.npy"), c_HBPM_Motif)
save_classifier_results(exportPath.format("s3d_motif_50.npy"), c_S3D_Motif)
save_classifier_results(exportPath.format("mmaction_motif_50.npy"), c_MMAction_Motif)

#%% ENS
print("VAME ENS")
c_VAME_ENS = classify(VAME_ENS)
print("DLC ENS")
c_DLC_ENS = classify(DLC_ENS)
print("hBPM ENS")
c_HBPM_ENS = classify(HBPM_ENS)
print("S3D ENS")
c_S3D_ENS = classify(S3D_ENS)
print("MMAction ENS")
c_MMAction_ENS = classify(MMAction_ENS)
print("=======================")
save_classifier_results(exportPath.format("vame_ens_50.npy"), c_VAME_ENS)
save_classifier_results(exportPath.format("dlc_ens_50.npy"), c_DLC_ENS)
save_classifier_results(exportPath.format("hbpm_ens_50.npy"), c_HBPM_ENS)
save_classifier_results(exportPath.format("s3d_ens_50.npy"), c_S3D_ENS)
save_classifier_results(exportPath.format("mmaction_ens_50.npy"), c_MMAction_ENS)

#%% ENS Diff
print("VAME ENS1")
c_VAME_ENS1= classify(VAME_ENS1)
print("DLC ENS1")
c_DLC_ENS1 = classify(DLC_ENS1)
print("hBPM ENS1")
c_HBPM_ENS1 = classify(HBPM_ENS1)
print("S3D ENS1")
c_S3D_ENS1 = classify(S3D_ENS1)
print("MMAction ENS1")
c_MMAction_ENS1 = classify(MMAction_ENS1)
print("=======================")
save_classifier_results(exportPath.format("vame_ens1_50.npy"), c_VAME_ENS1)
save_classifier_results(exportPath.format("dlc_ens1_50.npy"), c_DLC_ENS1)
save_classifier_results(exportPath.format("hbpm_ens1_50.npy"), c_HBPM_ENS1)
save_classifier_results(exportPath.format("s3d_ens1_50.npy"), c_S3D_ENS1)
save_classifier_results(exportPath.format("mmaction_ens1_50.npy"), c_MMAction_ENS1)

#%% VAME
print("VAME Entropy")
c_VAME_Entropy = classify(VAME_Entropy)
print("VAME Volume")
c_VAME_Volume = classify(VAME_Volume)
print("VAME Count")
c_VAME_Count = classify(VAME_Count)
print("=======================")
save_classifier_results(exportPath.format("vame_entropy_50.npy"), c_VAME_Entropy)
save_classifier_results(exportPath.format("vame_volume_50.npy"), c_VAME_Volume)
save_classifier_results(exportPath.format("vame_count_50.npy"), c_VAME_Count)


#%% Assessment + VAME
print("Assessment + VAME Motif")
c_VAME_AM = classify(VAME_AM)
print("Assessment + VAME ENS")
c_VAME_AENS = classify(VAME_AENS)
print("Assessment + VAME Entropy")
c_VAME_AE = classify(VAME_AE)
print("Assessment + VAME Volume")
c_VAME_AV = classify(VAME_AV)
print("Assessment + VAME Count")
c_VAME_AC = classify(VAME_AC)
print("=======================")
save_classifier_results(exportPath.format("am_50.npy"), c_VAME_AM)
save_classifier_results(exportPath.format("aens_50.npy"), c_VAME_AENS)
save_classifier_results(exportPath.format("ae_50.npy"), c_VAME_AE)
save_classifier_results(exportPath.format("av_50.npy"), c_VAME_AV)
save_classifier_results(exportPath.format("ac_50.npy"), c_VAME_AC)

#%% Assessment + VAME+
print("Assessment + VAME Motif + ENS")
c_VAME_AME = classify(VAME_AME)
print("Assessment + VAME Motif + ENS + Volume")
c_VAME_AMEV = classify(VAME_AMEV)
print("Assessment + VAME Motif + ENS + Volume + Count")
c_VAME_AMEVC = classify(VAME_AMEVC)
print("=======================")
save_classifier_results(exportPath.format("ame_50.npy"), c_VAME_AM)
save_classifier_results(exportPath.format("amev_50.npy"), c_VAME_AMEV)
save_classifier_results(exportPath.format("amevc_50.npy"), c_VAME_AMEVC)


#%% Graph classifier results
data = {'Assessment': a, 'VAME': vc, 'HBMP': hc, 'S3D': sc, 'MMAction': mc}
path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\Figures\Figure 5 - classification\models classification\count_3_splits_50'
graph_classifier_results(data, path)

data = {'Assessment': a, 'VAME': vc1, 'HBMP': hc1, 'S3D': sc1, 'MMAction': mc1}
path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\Figures\Figure 5 - classification\models classification\count_diff_3_1_50'
graph_classifier_results(data, path)

#%%
data = {'Assessment': a, 'Motif': m, 'ENS': ens, 'Entropy': e, 'Volume': v, 'Count': c}
path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\Figures\Figure 5 - classification\vames classsification\single_metric_50'
graph_classifier_results(path, data, 'Classication of Different Quantitative Methods')
save_classifier_results(path + '_results.npy', data)


#%%
# c_VAME_Motif
# c_DLC_Motif
# c_HBPM_Motif
# c_S3D_Motif
# c_MMAction_Motif

import scipy

#%% Motif
print("vame motif: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_Motif[0], axis=0))
print("dlc motif: ")
print(scipy.stats.f_oneway(c_assessment[0], c_DLC_Motif[0], axis=0))
print("hbpm motif: ")
print(scipy.stats.f_oneway(c_assessment[0], c_HBPM_Motif[0], axis=0))
print("s3d motif: ")
print(scipy.stats.f_oneway(c_assessment[0], c_S3D_Motif[0], axis=0))
print("mmaction motif: ")
print(scipy.stats.f_oneway(c_assessment[0], c_MMAction_Motif[0], axis=0))

#%% ENS
print("vame ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_ENS[0], axis=0))
print("dlc ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_DLC_ENS[0], axis=0))
print("hbpm ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_HBPM_ENS[0], axis=0))
print("s3d ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_S3D_ENS[0], axis=0))
print("mmaction ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_MMAction_ENS[0], axis=0))

#%% ENS1
print("vame ens1: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_ENS1[0], axis=0))
print("dlc ens1: ")
print(scipy.stats.f_oneway(c_assessment[0], c_DLC_ENS1[0], axis=0))
print("hbpm ens1: ")
print(scipy.stats.f_oneway(c_assessment[0], c_HBPM_ENS1[0], axis=0))
print("s3d ens1: ")
print(scipy.stats.f_oneway(c_assessment[0], c_S3D_ENS1[0], axis=0))
print("mmaction ens1: ")
print(scipy.stats.f_oneway(c_assessment[0], c_MMAction_ENS1[0], axis=0))

#%% VAME
print("vame motif: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_Motif[0], axis=0))
print("vame ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_ENS[0], axis=0))
print("vame ens1: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_ENS1[0], axis=0))
print("vame entropy: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_Entropy[0], axis=0))
print("vame volume: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_Volume[0], axis=0))
print("vame count: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_Count[0], axis=0))

#%% Assessment + VAME
print("assessment + vame motif: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AM[0], axis=0))
print("assessment + vame ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AENS[0], axis=0))
print("assessment + vame entropy: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AE[0], axis=0))
print("assessment + vame volume: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AV[0], axis=0))
print("assessment + vame count: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AC[0], axis=0))

#%% Assessment + VAME+
print("assessment + vame motif + ens: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AME[0], axis=0))
print("assessment + vame motif + ens + volume: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AMEV[0], axis=0))
print("assessment + vame motif + ens + volume + count: ")
print(scipy.stats.f_oneway(c_assessment[0], c_VAME_AMEVC[0], axis=0))


#%%
vEntropy = load_data(vEntropyPath, idf=BD)
hEntropy = load_data(hEntropyPath, idf=BD)
sEntropy = load_data(sEntropyPath, idf=BD)

vEntropy1 = epoch_sub(vEntropy, 3, 1)
hEntropy1 = epoch_sub(hEntropy, 3, 1)
sEntropy1 = epoch_sub(sEntropy, 3, 1)

vEntropy1 = combine_df(BD, vEntropy1)
hEntropy1 = combine_df(BD, hEntropy1)
sEntropy1 = combine_df(BD, sEntropy1)
#mEntropy = load_data(mEntropyPath, idf=BD)
#%%
vCount = load_data(vCountPath, idf=BD)
hCount = load_data(hCountPath, idf=BD)
sCount = load_data(sCountPath, idf=BD)
mCount = load_data(mCountPath, idf=BD)

vCount1 = epoch_sub(vCount, 3, 1)
hCount1 = epoch_sub(hCount, 3, 1)
sCount1 = epoch_sub(sCount, 3, 1)
mCount1 = epoch_sub(mCount, 3, 1)

vCount1 = combine_df(BD, vCount1)
hCount1 = combine_df(BD, hCount1)
sCount1 = combine_df(BD, sCount1)
mCount1 = combine_df(BD, mCount1)


#%% 
print("Assessment")
a = classify(assessment)
print("AM")
am = classify(VAME_AM)
print("AENS")
aens = classify(VAME_AENS)
print("AE")
ae = classify(VAME_AE)
print("AV")
av = classify(VAME_AV)
print("AC")
ac = classify(VAME_AC)

#%%
data = {'Assessment': a, 'A+Motif': am, 'A+ENS': aens, 'A+Entropy': ae, 'A+Volume': av, 'A+Count': ac}
path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\Figures\Figure 5 - classification\vames classsification\assessment_plus_metric_50'
graph_classifier_results(path, data, 'Classification with Assessment + Behavioral Data')

#%%
save_classifier_results(path + '_results.npy', data)
#%% Save results

save_classifier_results(exportPath.format("mmaction_motif_50.npy"), mm)
save_classifier_results(exportPath.format("vame_ens_50.npy"), ve)
save_classifier_results(exportPath.format("hbpm_ens_50.npy"), he)
save_classifier_results(exportPath.format("s3d_ens_50.npy"), se)
save_classifier_results(exportPath.format("mmaction_ens_50.npy"), me)

#%%
save_classifier_results(exportPath.format("assessments_scales_50.npy"), a)
save_classifier_results(exportPath.format("vame_motif_50.npy"), m)
save_classifier_results(exportPath.format("vame_ens_50.npy"), e)
save_classifier_results(exportPath.format("vame_am_50.npy"), am)
save_classifier_results(exportPath.format("vame_ae_50.npy"), ae)
save_classifier_results(exportPath.format("vame_ame_50.npy"), ame)


#%% Graph classifier results
data = {'Assessment': a, 'Motif': m, 'ENS': e, 'AM': am, 'AE': ae, 'AME': ame}
graph_classifier_results(data)


ymrsPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\YMRS.csv"
hamdPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\HAMD.csv"

#%%
VAME_ENS = load_data(vENSPath)
VAME_ENS1 = epoch_sub(VAME_ENS, 2, 0)


YMRS = load_data(ymrsPath, idf=VAME_ENS1)
HAMD = load_data(hamdPath, idf=VAME_ENS1)

#%% 
print("YMRS")
y = classify(YMRS)
print("HAMD")
h = classify(HAMD)
# %%
