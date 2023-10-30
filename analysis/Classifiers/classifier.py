#%%
import csv
import numpy as np
import pandas as pd
import os
import seaborn as sns 
import matplotlib.pyplot as plt
import random

# %%
# https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
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
dataframe = {}

#%% diagnostic scales
"""

ASSESSMENT SCALES START
"""
diagnosticPath = r"C:\Users\kietc\SURF\jack-data\scaled_diagnostic_data.csv"

with open(diagnosticPath, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        dataframe[row[0]] = [int(row[1])]
        vec = np.array(row[2:4]).astype(float).tolist()
        vec = [i * 30 for i in vec]
        dataframe[row[0]].extend(vec)

"""
DIAGNOSTIC SCALES END
"""
"""
VAME START
"""

#%% VAME motif usages overall
# 10 motifs = 10 columns
motifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"

with open(motifPath, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        vec = np.array(row[1:]).astype(float).tolist()
        vec = [x/27000 for x in vec ] # scaled
        dataframe[row[0]].extend(vec)

#%% VAME motif usages in 30 sec intervals
# 10 motifs x 30 intervals = 300 columns
motifPath = r"C:\Users\kietc\SURF\jack-data\motif_usage_30s_interval.csv"

with open(motifPath, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        vec = np.array(row[1:]).astype(float).tolist()
        dataframe[row[0]].extend(vec)

#%% VAME transition matrix
# 10 motifs x 10 motifs = 100 columns
transitionPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix\{}.npy"
for v in videos:
    transition = np.load(transitionPath.format(v)).flatten()
    dataframe[v].extend(transition)

"""
VAME END
"""
"""
hBPM START
"""
#%% hBPM motif usages overall
# 10 motifs = 10 columns
motifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"

with open(motifPath, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        vec = np.array(row[1:]).astype(float).tolist()
        vec = [x/27000 for x in vec ] # scaled
        dataframe[row[0]].extend(vec)

#%% hBPM motif usages in 30 sec intervals

# 11 motifs x 30 intervals = 330 columns
motifPath = r"C:\Users\kietc\SURF\jack-data\hBPM_motif_usage_30s_interval.csv"

with open(motifPath, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        vec = np.array(row[1:]).astype(float).tolist()
        dataframe[row[0]].extend(vec)

#%% hBPM transition matrix
# 11 motifs x 11 motifs = 121 columns
transitionPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\transition_matrix\{}.npy"
for v in videos:
    transition = np.load(transitionPath.format(v)).flatten()
    dataframe[v].extend(transition)

"""
hBPM END
"""
#%%
X = []
Y = []
seeds = []
for video in dataframe:
    X.append(dataframe[video][1:])
    Y.append(dataframe[video][0])

# %%
acc = []
pre = []
rec = []

rand = random.randrange(100)
for i in range(rand, rand + 1):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.24, stratify=Y, random_state=i)

    # x
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(xtrain)
    xtest = sc_x.transform(xtest)

    # model
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(xtrain, ytrain)
    y_pred = classifier.predict(xtest)

    nX = np.array(xtest)
    nY = np.array(ytest)

    x_min, x_max = nX[:, 0].min() - 1, nX[:, 0].max() + 1
    y_min, y_max = nX[:, 1].min() - 1, nX[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(nX[:, 0], nX[:, 1], c=nY, marker='o', edgecolor='k')
    plt.xlabel('YMRS')
    plt.ylabel('HAMD')
    plt.show()
   
    acc.append(accuracy_score(ytest, y_pred))
    pre.append(precision_score(ytest, y_pred))
    rec.append(recall_score(ytest, y_pred))

    # cross validation
    scoring = ['accuracy', 'precision', 'recall']
    scores = cross_validate(classifier, xtrain, ytrain, scoring = scoring, cv = 3)
    # print('Accuracy: ', scores['test_accuracy'], 'mean:',  scores['test_accuracy'].mean())
    # print('Precision:', scores['test_precision'], 'mean:',  scores['test_precision'].mean())
    # print('Recall:', scores['test_recall'], 'mean:',  scores['test_recall'].mean())
    
    acc.extend(scores['test_accuracy'])
    pre.extend(scores['test_precision'])
    rec.extend(scores['test_recall'])
    seeds.append(i)

print('Accuracy: %.05f' % np.mean(acc))
print('Precision: %.05f' % np.mean(pre))
print('Recall: %.05f' % np.mean(rec))

#%%
def export(input):
    exportPath =  r"C:\Users\kietc\OneDrive - UC San Diego\SURF\Classification"
    labels = ['Accuracy', 'Precision', 'Recall']

    data = [acc, pre, rec]

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

    plt.savefig(os.path.join(exportPath, fname), transparent=True)
    plt.savefig(os.path.join(exportPath, fname_pdf), transparent=True)

    np.save(os.path.join(exportPath, input + '_acc_50_tests.npy'), np.array(acc))
    np.save(os.path.join(exportPath, input + '_pre_50_tests.npy'), np.array(pre))
    np.save(os.path.join(exportPath, input + '_rec_50_tests.npy'), np.array(rec))

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


 # confusion matrix  
    # cm = confusion_matrix(ytest, y_pred)
    # print ("Confusion Matrix : \n", cm)
    # class_names=[0,1] # name  of classes 

    # https://www.projectpro.io/recipes/perform-logistic-regression-sklearn
    # sns.set(font_scale=2)
    # fig, ax = plt.subplots() 
    # tick_marks = np.arange(len(class_names)) 
    # plt.xticks(tick_marks, class_names) 
    # plt.yticks(tick_marks, class_names) 

    # create heatmap 
    # sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g') 
    # ax.xaxis.set_label_position("top") 
    # plt.tight_layout() 
    # plt.title('Confusion matrix', y=1.1) 
    # plt.ylabel('Actual label') 
    # plt.xlabel('Predicted label')

    # print
    # print("Accuracy:", accuracy_score(ytest, y_pred)) 
    # print("Precision:", precision_score(ytest, y_pred)) 
    # print("Recall:", recall_score(ytest, y_pred))
