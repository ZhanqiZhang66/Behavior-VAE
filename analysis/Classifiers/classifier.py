#%%
import csv
import numpy as np
import pandas as pd
import os
import seaborn as sns 
import matplotlib.pyplot as plt
import random

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
def loadDataframe(path, scores = False, gender = False, scaled = False):
    dataframe = {}
    with open(path, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        e = 5 if gender else 4
        next(csvreader)
        for row in csvreader:
            dataframe[row[0]] = [int(row[1])]
            if scores:
                vec = np.array(row[2:e]).astype(float).tolist()
            if scaled:
                vec = [i * 30 for i in vec]
            if scores:
                dataframe[row[0]].extend(vec)
    return dataframe


def loadMotifUsage(path, diagnosticPath, scaled = False):
    dataframe = loadDataframe(diagnosticPath)
    with open(path, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        next(csvreader)
        for row in csvreader:
            vec = np.array(row[1:]).astype(float).tolist()
            if scaled:
                vec = [x/27000 for x in vec]
            dataframe[row[0]].extend(vec)
    return dataframe

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

        # nX = np.array(xtest)
        # nY = np.array(ytest)

        # x_min, x_max = nX[:, 0].min() - 1, nX[:, 0].max() + 1
        # y_min, y_max = nX[:, 1].min() - 1, nX[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)

        # plt.contourf(xx, yy, Z, alpha=0.4)
        # plt.scatter(nX[:, 0], nX[:, 1], c=nY, marker='o', edgecolor='k')
        # plt.xlabel('YMRS')
        # plt.ylabel('HAMD')
        # plt.show()
    
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

    print('Accuracy: %.05f' % np.mean(acc))
    print('Precision: %.05f' % np.mean(pre))
    print('Recall: %.05f' % np.mean(rec))
    data = [acc, pre, rec]
    return data

def export(path, input, data):
    labels = ['Accuracy', 'Precision', 'Recall']

    # Create a box plot
    bp = plt.boxplot(data, labels=labels, showmeans=True, meanline=True)

    # Set the title and labels
    plt.title(input)

    # Display the mean values as markers
    for i, line in enumerate(bp['medians']):
        x, y = line.get_data()
        mean_value = np.mean(data[i])
        plt.text(x[0] + 0.15, y[1] + 0.01, f'{mean_value:.3f}', horizontalalignment='center', verticalalignment='bottom', fontsize=12)

    fname = "{}_50_tests.png".format(input)
    fname_pdf = "{}_50_tests.pdf".format(input)

    plt.savefig(os.path.join(path, fname), transparent=True)
    plt.savefig(os.path.join(path, fname_pdf), transparent=True)

    np.save(os.path.join(path, input + '_50_tests.npy'), np.array([data[0], data[1], data[2]]))

#%%
diagnosticPath = r"C:\Users\kietc\SURF\jack-data\scaled_diagnostic_data.csv"
VAMEMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"
hBPMMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
S3DMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
MMActionMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"


exportPath =  r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification"

#%%
assessment = loadDataframe(diagnosticPath, scores=True)
vame = loadMotifUsage(VAMEMotifPath, diagnosticPath, True)
hBPM = loadMotifUsage(hBPMMotifPath, diagnosticPath, True)
S3D = loadMotifUsage(S3DMotifPath, diagnosticPath, True)
MMAction = loadMotifUsage(MMActionMotifPath, diagnosticPath, True)


#%%
print("Assessment")
a = classify(assessment)
print("VAME")
v = classify(vame)
print("hBPM")
h = classify(hBPM)
print("S3D")
s = classify(S3D)
print("MMAction")
m = classify(MMAction)

#%%
data_sets = {'Assessment': a, 'VAME': v, 'hBPM': h, 'S3D': s, '': m}
name = ['a', 'v', 'h', 's', 'm']
df_list = []
for data_name, data_set in data_sets.items():
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
plt.title('Motif Usage')

# Show the plot
plt.show()


#%%
export("Diagnostic_scales")
#%%
export("VAME_motif_usage")
#%%
export("hBPM_motif_usage")
#%%
export("VAME_transition_matrix")
#%%
export("hBPM_transition_matrix")




#%%
# #%% VAME motif usages in 30 sec intervals
# # 10 motifs x 30 intervals = 300 columns
# motifPath = r"C:\Users\kietc\SURF\jack-data\motif_usage_30s_interval.csv"

# with open(motifPath, 'r') as file:
#     csvreader = csv.reader(file, delimiter=',')
#     next(csvreader)
#     for row in csvreader:
#         vec = np.array(row[1:]).astype(float).tolist()
#         dataframe[row[0]].extend(vec)

# #%% VAME transition matrix
# # 10 motifs x 10 motifs = 100 columns
# transitionPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix\{}.npy"
# for v in videos:
#     transition = np.load(transitionPath.format(v)).flatten()
#     dataframe[v].extend(transition)

# #%% hBPM motif usages in 30 sec intervals

# # 11 motifs x 30 intervals = 330 columns
# motifPath = r"C:\Users\kietc\SURF\jack-data\hBPM_motif_usage_30s_interval.csv"

# with open(motifPath, 'r') as file:
#     csvreader = csv.reader(file, delimiter=',')
#     next(csvreader)
#     for row in csvreader:
#         vec = np.array(row[1:]).astype(float).tolist()
#         dataframe[row[0]].extend(vec)

# #%% hBPM transition matrix
# # 11 motifs x 11 motifs = 121 columns
# transitionPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\transition_matrix\{}.npy"
# for v in videos:
#     transition = np.load(transitionPath.format(v)).flatten()
#     dataframe[v].extend(transition)

