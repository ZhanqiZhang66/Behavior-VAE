# %%
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


# %%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", 
              "BC1ANBU", "BC1ANGA", "BC1ANHE", "BC1ANWI", "BC1ASKA", 
              "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", 
              "BC1CISI", "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", 
              "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", "BC1HETR", 
              "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", 
              "BC1LABO", "BC1LACA", "BC1LESA", "BC1LOKE", "BC1LOMI", 
              "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
              "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", 
              "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

# %% normalize actions for all videos
for i in videos:
    src_path = '../../data/S3D/time/' + i + '_Actions.csv'
    out_path = '../../data/S3D/percentage/' + i + '_percentages.csv'

    actions = {}
    total_time = 0
    
    with open(src_path, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        
        for row in csvreader:
            actions[row[0]] = float(row[1])
            total_time += float(row[1])
        
        with open(out_path,"w", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for key in actions:
                csvwriter.writerow((key, actions[key]/total_time))

# %% get actions
actions = []
for i in videos:
    path = '../../data/S3D/percentage/' + i + '_percentages.csv'
    with open(path, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        for row in csvreader:
            action = row[0]
            if action not in actions: 
                actions.append(action)
actions.insert(0, "")
print(actions)


# %% generate csv
out_file = '../../data/S3D/s3d_data.csv'
with open(out_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, lineterminator = '\n')
    csvwriter.writerow(actions)
    for i in videos:
        path = '../../data/S3D/percentage/' + i + '_percentages.csv'
        vec = ["0" for x in range(len(actions))]
        vec[0] = i
        with open(path, 'r') as file:
            csvreader = csv.reader(file, delimiter=',')
            for row in csvreader:
                vec[actions.index(row[0])] = row[1]
        csvwriter.writerow(vec)

#%%
path = '../../data/S3D/s3d_data.csv'

#%%
X = []
Y = []
with open(path, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        G = 0
        # 1-15 - s3d
        # 18 - gender
        if row[18] == 'F': G = 1
        vec = np.array(row[1:18]).astype(float).tolist()
        vec.append(G)
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
