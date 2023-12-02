#%% 
import csv
import os
import numpy as np
from collections import Counter
from utils import loadLabels
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#%%
def generateMotifUsage(labels, split, motifSize, scaled, missing):
    frames = len(labels[next(iter(labels))])
    motifUsage = {}
    for v in videos:
        motifUsage[v] = []
        for i in range(split):
            currUsage = [0] * motifSize
            for j in range(i * frames // split, (i + 1) * frames // split):
                if missing and labels[v][j] < 0:
                    currUsage[motifSize-1] += 1
                elif labels[v][j] < 0:
                    continue
                else:
                    currUsage[labels[v][j]] += 1
            if scaled:
                currUsage = [x / (frames // split) for x in currUsage]
            motifUsage[v].append(currUsage)
    return motifUsage

def saveMotifUsage(path, motifUsage, split, motifSize, missing=False):
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


def graphMotifUsage(path, motifUsage, BD, motifSize, title, top=0, wmulti=6):
    population = ["BD" if v in BD else "HC" for v in videos]
    population = [item for item in population for _ in range(motifSize)]

    motif = [f'Motif {i}' for i in range(motifSize)] * 50

    value = [motifUsage[v][0] for v in videos]
    value = [motif for v in value for motif in v]
    value = [motif / 27000 for motif in value]

    print(len(population), len(motif), len(value))

    df = pd.DataFrame({'Population': population,
                    'Motif': motif,
                    'Values': value})

    if top > 0:
        mean = df.groupby('Motif')['Values'].mean()
        sorted_motifs = mean.sort_values(ascending=False)
        top_motifs = sorted_motifs.head(10)
        df = df[df['Motif'].isin(top_motifs.index)]
        motifSize = top

    w = motifSize/10 * wmulti
    fig, ax = plt.subplots(1, 1, figsize=(w, 4))
    violin = sns.boxplot(x='Motif', y='Values', hue='Population', data=df)
    violin.legend_.remove()
    #handles = violin.legend_.legendHandles
    # ax.legend(handles, labels)
    plt.xlabel('Motif')
    plt.ylabel('Values')
    ax.set_title(title)

    plt.show()
    fname = "motif_usage_{}.png".format(title)
    fname_pdf = "motif_usage_{}.pdf".format(title)
    fig.savefig(os.path.join(path, fname), transparent=True)
    fig.savefig(os.path.join(path, fname_pdf), transparent=True)


#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]
BD = ["BC1ADPI", "BC1BRBU", "BC1CISI", "BC1DOBO", "BC1JACL",
    "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", "BC1LESA", 
    "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", 
    "BC1MEMA", "BC1MISE", "BC1OKBA", "BC1REFU", "CASH1",
    "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]
 

#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", 'BC1BRBU']
BD = ["BC1ADPI", 'BC1BRBU']

#%%
vLabelPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
vMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_overall.csv"
vFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\classification"

#%%
inPath  = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
outPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
figPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\classification"

#%%
inPath  = r'C:\Users\kietc\SURF\jack-data\S3D\s3d_labels\s3d_labels_{}.npy'
outPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
figPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\classification"


#%%
inPath  = r'C:\Users\kietc\SURF\jack-data\MMAction\mmaction_labels\mmaction_labels_{}.npy'
outPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"
figPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\classification"

#%%
frames = 27000
split = 1
motifSize = 81
#81, 401

# %%
labels = loadLabels(inPath, videos, frames)
motifUsage = generateMotifUsage(labels, split, motifSize, False, True)
graphMotifUsage(figPath, motifUsage, BD, motifSize, 'hBPM')

# %%
labels = loadLabels(inPath, videos, frames)
motifUsage = generateMotifUsage(labels, split, motifSize, False, True)
graphMotifUsage(figPath, motifUsage, BD, motifSize, 'MMAction', top=10, wmulti=9)


# %%
saveMotifUsage(outPath, motifUsage, split, motifSize)


#%%
population = ["BD" if v in BD else "HC" for v in videos]
population = [item for item in population for _ in range(motifSize)]

motif = [f'Motif {i}' for i in range(motifSize)] * 50

value = [motifUsage[v][0] for v in videos]
value = [motif for v in value for motif in v]
value = [motif / 27000 for motif in value]

print(len(population), len(motif), len(value))

df = pd.DataFrame({'Population': population,
                'Motif': motif,
                'Values': value})


# %%
