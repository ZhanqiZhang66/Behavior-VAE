#%%
import csv
import numpy as np
import pandas as pd
import os
import seaborn as sns 
import matplotlib.pyplot as plt

# %%
# https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate


#%%
path = '../../data/dv_data.csv'

#%%
X = []
Y = []
with open(path, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        G = 0
        if row[117] == 'F': G = 1
        # motif - 1:10
        # transition - 11:110
        # entropy - 111
        # num_zero_row - 112
        # num_one_item - 113
        # num_zero_item - 114
        # YMRS - 115
        # HAMD - 116
        # Gender - 117
        vec = np.array(row[1:117]).astype(float).tolist()
        #vec.extend(np.array(row[111:112]).astype(float).tolist())
        #vec.extend(np.array(row[115:117]).astype(float).tolist())
        #vec.append(G)
        X.append(vec)
        Y.append(int(row[-1]))

print(X)
print(Y)

# %%
acc = []
pre = []
rec = []

for i in range(1):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.24, stratify=Y, random_state=i)

    # x
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(xtrain)
    xtest = sc_x.transform(xtest)

    # model
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(xtrain, ytrain)
    y_pred = classifier.predict(xtest)

    # confusion matrix  
    cm = confusion_matrix(ytest, y_pred)
    # print ("Confusion Matrix : \n", cm)
    class_names=[0,1] # name  of classes 

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

print('Accuracy:', np.mean(acc))
print('Precision:', np.mean(pre))
print('Recall:', np.mean(rec))
# %%
