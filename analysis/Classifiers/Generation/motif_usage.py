#%% 
import csv
import os
import numpy as np
from collections import Counter
from utils import loadLabels


#%%
def generateMotifUsage(labels, split, motifSize, scaled):
    frames = len(labels[next(iter(labels))])
    motifUsage = {}
    for v in videos:
        motifUsage[v] = []
        for i in range(split):
            currUsage = [0] * motifSize
            for j in range(i * frames // split, (i + 1) * frames // split):
                currUsage[labels[v][j]] += 1
            if scaled:
                currUsage = [x / (frames // split) for x in currUsage]
            motifUsage[v].append(currUsage)
    return motifUsage

#%%
def saveMotifUsage(path, motifUsage, split, motifSize):
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator="\n")
        header = ['video']
        for i in range(split):
            for j in range(motifSize):
                header.append(str(i) + 'm' + str(j))
        csvwriter.writerow(header)

        for v in videos:
            row = [v]
            for i in range(split):
                row.extend(motifUsage[v][i])
            csvwriter.writerow(row)


#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA"]

#%%
inPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
outPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"

#%%
inPath  = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
outPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"

#%%
inPath  = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
outPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"

#%%
frames = 27000
split = 1
motifSize = 11

# %%
labels = loadLabels(inPath, videos, frames)
motifUsage = generateMotifUsage(labels, split, motifSize, False)

# %%
saveMotifUsage(outPath, motifUsage, split, motifSize)
# %%
