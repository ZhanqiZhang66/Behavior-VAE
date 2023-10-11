#%% 
import csv
import os
import numpy as np

#%% config
path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]


#%% Check if video folder exists
found = 0
for v in videos:
    if os.path.isdir(path.format(v, v)):
        found += 1
    else:
        print(path.format(v, v))
print(found)

#%% Retrieve and truncate 10_km_labels

labels = {}
minLength = float('inf')

for v in videos:
    labels[v] = np.load(path.format(v, v))
    minLength = min(minLength, labels[v].shape[0])

# truncating - max difference is 2200 frames
for v in videos:
    labels[v] = labels[v][:minLength]

#%% motif_usage of 30s interval
minLength = minLength//900*900
motifUsage = {}
for v in videos:
    motifUsage[v] = []
    for i in range(0, minLength, 900):
        currUsage = [0]*10
        for j in range(i, i + 900):
            currUsage[labels[v][j]] += 1
        currUsage = [x / 900 for x in currUsage]
        motifUsage[v].append(currUsage)

#%% Generate motif_usage data
outFile = r"C:\Users\kietc\SURF\jack-data\motif_usage_30s_interval.csv"
header = ["video"]
for i in range(0, 10):
    for j in range(0, minLength, 900):
        header.append("{}-{}".format(i, j))
        
with open(outFile, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, lineterminator="\n")
    csvwriter.writerow(header)

    for v in videos:
        row = [v]
        motif = motifUsage[v]
        for i in range(len(motif[0])):
            for j in range(len(motif)):
                row.append(motif[j][i])
        csvwriter.writerow(row)

# %%
