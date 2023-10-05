#%%
import csv
import numpy as np
import pandas as pd
import os
import seaborn as sns 
import matplotlib.pyplot as plt

#%% Get actions
import glob

path = '../../data/stdet'
csv_files = glob.glob(os.path.join(path, "*.csv"))

actions = []
for f in csv_files:
    with open(f, 'r') as file:
        csvreader = csv.reader(file, delimiter=',')
        next(csvreader)
        for row in csvreader:
            for i in row:
                if len(i.split("'")) > 1:
                    action = i.split("'")[1]
                    if action not in actions: actions.append(action)
actions.insert(0, "")
print(actions)

#%% Generate mm_data.csv
path = '../../data/stdet'
csv_files = glob.glob(os.path.join(path, "*.csv"))

out_file = "../../data/mm_data.csv"
    
# writing to csv file 
with open(out_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, lineterminator = '\n')
    csvwriter.writerow(actions)
    for f in csv_files:
        curr_actions = {}
        with open(f, 'r') as file:
            csvreader = csv.reader(file, delimiter=',')
            next(csvreader)
            for row in csvreader:
                for i in row:
                    if len(i.split("'")) > 1:
                        action = i.split("'")[1]
                        prob = float(i.split("'")[2][2:i.split("'")[2].rindex(')')])
                        if curr_actions.get(action) == None: curr_actions[action] = 0
                        curr_actions[action] += prob
        vec = [0] * (len(actions))
        vec[0] = f[17:-4]
        for i in curr_actions:
            vec[actions.index(i)] = curr_actions.get(i)
        csvwriter.writerow(vec)

#%%
path = '../../data/mm_data.csv'

#%%
X = []
Y = []
with open(path, 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        G = 0
        if row[22] == 'F': G = 1
        vec = np.array(row[1:21]).astype(float).tolist()
        vec.append(G)
        X.append(vec)
        Y.append(int(row[-1]))

# %%
# https://www.geeksforgeeks.org/ml-logistic-regression-using-python/
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=0)

# %%
from sklearn.preprocessing import StandardScaler
  
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

# %%
from sklearn.linear_model import LogisticRegression
  
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)

# %%
from sklearn.metrics import confusion_matrix
  
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)
class_names=[0,1] # name  of classes 

# https://www.projectpro.io/recipes/perform-logistic-regression-sklearn
sns.set(font_scale=2)
fig, ax = plt.subplots() 
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 
plt.yticks(tick_marks, class_names) 

# create heatmap 
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top") 
plt.tight_layout() 
plt.title('Confusion matrix', y=1.1) 
plt.ylabel('Actual label') 
plt.xlabel('Predicted label')

# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Accuracy:", accuracy_score(ytest, y_pred)) 

print("Precision:", precision_score(ytest, y_pred)) 

print("Recall:", recall_score(ytest, y_pred)) 
