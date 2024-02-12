#%% 
import csv
import os
import numpy as np
from collections import Counter
from utils import load_motif_labels
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


def graphMotifUsage(path, motifUsage, BD, motifSize, title, top=0, wmulti=6, labelMap=None):
    population = ["BD" if v in BD else "HC" for v in videos]
    population = [item for item in population for _ in range(motifSize)]

    motif = [f'Motif {i}' for i in range(motifSize)] * 50
    if labelMap:
        motif = [f'{labelMap[i]}' for i in range(motifSize)] * 50
    print(motif)
    value = [motifUsage[v][0] for v in videos]
    value = [motif for v in value for motif in v]
    #value = [motif / 27000 for motif in value]

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

def loadKineticsMap(path):
    map = {0: "none"}
    with open(path, 'r') as file:
        lines = file.readlines()
        i = 1
        for line in lines:
            map[i] = line
            i += 1
    map[i] = "none"
    return map

def loadAvaMap(path):
    map = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            i = int(line.split(":")[0])
            map[i-1] = line.split(": ")[1]
    map[i] = "none"
    for j in range(1, i):
        if j not in map:
            map[j] = "none"
    return map



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
v3MotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\motif_usage_3_split.csv"
vFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\classification"

#%%
hLabelPath  = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
hMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\motif_usage_overall.csv"
hFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\classification"

#%%
sLabelPath  = r'C:\Users\kietc\SURF\jack-data\S3D\s3d_labels\s3d_labels_{}.npy'
sMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\motif_usage_overall.csv"
sFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\Figures\Figure 5 - classification\motif usage"

#%%
mLabelPath  = r'C:\Users\kietc\SURF\jack-data\MMAction\mmaction_labels\mmaction_labels_{}.npy'
mMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\motif_usage_overall.csv"
mFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\Figures\Figure 5 - classification\motif usage"

#%%
dLabelPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\DLC_10_km_label_{}.npy'
dMotifPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\motif_usage_overall.csv"
dFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\classification"

# %%
avaPath = r'C:\Users\kietc\SURF\Behavior-VAE\analysis\Classifiers\ava_80_label_map.txt'
avaMap = loadAvaMap(avaPath)
# %%
k400Path = r'C:\Users\kietc\SURF\Behavior-VAE\analysis\Classifiers\kinetics_400_label_map.txt'
k400Map = loadKineticsMap(k400Path)

# %%
vLabels = load_motif_labels(vLabelPath, videos, 27000)
vMotifUsage = generateMotifUsage(vLabels, 1, 10, scaled=True, missing=False)

# %%
dLabels = load_motif_labels(dLabelPath, videos, 27000)
dMotifUsage = generateMotifUsage(dLabels, 1, 10, scaled=True, missing=False)

# %% path, motifUsage, split, motifSize, missing=False
saveMotifUsage(dMotifPath, dMotifUsage, 1, 10)
# %%
graphMotifUsage(path=vFigPath, motifUsage=vMotifUsage, BD=BD, 
                motifSize=10, title='VAME', top=0, wmulti=7, labelMap=None)

# %%
sLabels = load_motif_labels(sLabelPath, videos, 27000)
sMotifUsage = generateMotifUsage(sLabels, 1, 401, scaled=True, missing=True)
# %%
graphMotifUsage(path=sFigPath, motifUsage=sMotifUsage, BD=BD, 
                motifSize=401, title='S3D', top=10, wmulti=20, labelMap=k400Map)
















#%%
mLabels = load_motif_labels(mLabelPath, videos, 27000)
mMotifUsage = generateMotifUsage(mLabels, 1, 81, scaled=True, missing=True)
# %%
graphMotifUsage(path=mFigPath, motifUsage=mMotifUsage, BD=BD, 
                motifSize=81, title='MMAction', top=10, wmulti=20, labelMap=avaMap)

# %%
sumUsage = [0] * 81
for v in videos:
    for i in range(81):
        sumUsage[i] += mMotifUsage[v][0][i]

sumUsage = [item / 50 for item in sumUsage]

# %%
# Get indices and values of the top 3 highest values
top_indices_values = sorted(enumerate(sumUsage), key=lambda x: x[1], reverse=True)[:3]

# Extract the indices and values from the result
top_indices, top_values = zip(*top_indices_values)

print("Indices of top 3 highest values:", top_indices)
print("Values of top 3 highest values:", top_values)


# %%
HP = [key for key in videos if key not in BD]


#%%HP
# Extract arrays from the dictionary values
arrays = [np.array(sMotifUsage[key][0]) for key in HP]

# Calculate the median array
median_array = np.median(arrays, axis=0)

# Calculate the interquartile range (IQR)
q75, q25 = np.percentile(arrays, [75 ,25], axis=0)
iqr = q75 - q25

# %%
sorted_indices = np.argsort(median_array)[::-1]
top3_values = median_array[sorted_indices[:3]]
top3_indices = sorted_indices[:3]

# %%
print(top3_values)
print(top3_indices)
print(iqr[top3_indices[0]], iqr[top3_indices[1]], iqr[top3_indices[2]])
# %%
