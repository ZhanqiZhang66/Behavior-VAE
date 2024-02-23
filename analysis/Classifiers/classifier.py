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

def epoch_sub(idf, indexA, indexB):
    df = {}
    for v in videos:
        df[v] = [(abs(idf[v][indexA]+1 - idf[v][indexB]+1)) / min(idf[v][indexA]+1,idf[v][indexB]+1)]
    return df

def classify(dataframe):
    random.seed(10)
    X = []
    Y = []
    seeds = []
    for video in dataframe:
        X.append(dataframe[video][1:])
        Y.append(dataframe[video][0])

    acc = []
    pre = []
    rec = []

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

    print('--Accuracy: %.05f (%.05f)' % (np.mean(acc), np.std(acc)))
    print('--Precision: %.05f (%.05f)' % (np.mean(pre), np.std(pre)))
    print('--Recall: %.05f (%.05f)' % (np.mean(rec), np.std(rec)))
    data = [acc, pre, rec]
    return data

def save_classifier_results(path, data):
    print(path)
    np.save(path, np.array(data))

def load_result(path):
    return np.load(path).tolist()

def graph_classifier_results(data):
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
    plt.title('VAME')

    # Show the plot
    plt.show()

#%% v - vame, h - hBPM, s - S3D, m - MMAction
if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'

    diagnosticPath = r"C:\Users\zhanq\OneDrive - UC San Diego\SURF\scaled_diagnostic_data.csv"

    VAMEMotifPath = r"C:\Users\zhanq\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"
    HBPMMotifPath = r"C:\Users\zhanq\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
    S3DMotifPath = r"C:\Users\zhanq\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
    MMActionMotifPath = r"C:\Users\zhanq\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"


    VAMEENSPath = r'C:\Users\zhanq\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
    HBPMENSPath = r'C:\Users\zhanq\OneDrive - UC San Diego\SURF\hBPM\entropy_3_spli.csv'
    S3DENSPath = r'C:\Users\zhanq\OneDrive - UC San Diego\SURF\S3D\entropy_3_spli.csv'
    MMActionENSPath = r'C:\Users\zhanq\OneDrive - UC San Diego\SURF\MMAction\entropy_3_spli.csv'

    exportPath = r"C:\Users\zhanq\OneDrive - UC San Diego\SURF\Classification\{}"
    exportResultPath = r"C:\Users\zhanq\OneDrive - UC San Diego\SURF\Classification\{}"

elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
else:
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
    diagnosticPath = r"C:\Users\kietc\SURF\jack-data\scaled_diagnostic_data.csv"

    VAMEMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"
    HBPMMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
    S3DMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
    MMActionMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"

    VAMEENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
    HBPMENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\entropy_3_spli.csv'
    S3DENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\entropy_3_spli.csv'
    MMActionENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\entropy_3_spli.csv'

    exportPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification\{}"
    exportResultPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification\{}"

    diagnosticPath = r"C:\Users\kietc\SURF\jack-data\scaled_diagnostic_data.csv"

    vMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"
    hMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
    sMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
    mMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"

    vEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
    hEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\entropy_3_split.csv'
    sEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\entropy_3_split.csv'
    mEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\entropy_3_split.csv'

    vENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\ens_3_split.csv'
    hENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\ens_3_split.csv'
    sENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\ens_3_split.csv'
    mENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\ens_3_split.csv'

    vVolumePath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\volume.csv"


#%%
volume = generate_volume(vVolumePath)

#%% Create DF for each model
BD = load_data(diagnosticPath, end=2)
assessment = load_data(diagnosticPath, end=-1)


#%%
VAME_Motif = load_data(vMotifPath, idf=BD, scale=27000)
HBPM_Motif = load_data(hMotifPath, idf=BD, scale=27000)
S3D_Motif = load_data(sMotifPath, idf=BD, scale=27000)
MMAction_Motif = load_data(mMotifPath, idf=BD, scale=27000)

#%%
VAME_ENS = load_data(VAMEENSPath, idf=BD)
HBPM_ENS = load_data(HBPMENSPath, idf=BD)
S3D_ENS = load_data(S3DENSPath, idf=BD)
MMAction_ENS = load_data(MMActionENSPath, idf=BD)

#%% ENS epoch 3 - epoch 1
VAME_ENS1 = epoch_sub(VAME_ENS, 3, 1)
VAME_ENS1 = combine_df(BD, VAME_ENS1)
HBPM_ENS1 = epoch_sub(HBPM_ENS, 3, 1)
HBPM_ENS1  = combine_df(BD, HBPM_ENS1)
S3D_ENS1 = epoch_sub(S3D_ENS, 3, 1)
S3D_ENS1 = combine_df(BD, S3D_ENS1)
MMAction_ENS1 = epoch_sub(MMAction_ENS, 3, 1)
MMAction_ENS1 = combine_df(BD, MMAction_ENS1)


#%% VAME Comparisons
VAME_Motif = load_data(vMotifPath, idf=BD, scale=27000)
VAME_ENS = load_data(VAMEENSPath, idf=BD)
VAME_Volume = load_data(vVolumePath, idf=BD)
VAME_AM = load_data(vMotifPath, idf=assessment, scale=27000)
VAME_AE = load_data(VAMEENSPath, idf=assessment)
VAME_AME = load_data(VAMEENSPath, idf=VAME_AM)

#%% Classify
print("Assessment")
a = classify(assessment)
print("VAME Motif")
vm = classify(VAME_Motif)
print("hBPM Motif")
hm = classify(HBPM_Motif)
print("S3D Motif")
sm = classify(S3D_Motif)
print("MMAction Motif")
mm = classify(MMAction_Motif)

#%%
print("VAME ENS")
ve = classify(VAME_ENS)
print("hBPM ENS")
he = classify(HBPM_ENS)
print("S3D ENS")
se = classify(S3D_ENS)
print("MMAction ENS")
me = classify(MMAction_ENS)

#%%
print("VAME ENS")
ve = classify(VAMEENS1)
print("hBPM ENS")
he = classify(HBPMENS1)
print("S3D ENS")
se = classify(S3DENS1)
print("MMAction ENS")
me = classify(MMActionENS1)

#%% 
print("Assessment")
a = classify(assessment)
print("VAME Motif")
m = classify(VAME_Motif)
print("VAME ENS")
e = classify(VAME_ENS)
print("VAME Volume")
v = classify(VAME_Volume)
print("VAME Motif + Assessment")
am = classify(VAME_AM)
print("VAME ENS + Assessment")
ae = classify(VAME_AE)
print("VAME Motif + ENS + Assessment")
ame = classify(VAME_AME)


#%% Save results
save_classifier_results(exportPath.format("assessments_scales_50.npy"), a)
save_classifier_results(exportPath.format("vame_motif_50.npy"), vm)
save_classifier_results(exportPath.format("hbpm_motif_50.npy"), hm)
save_classifier_results(exportPath.format("s3d_motif_50.npy"), sm)
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
