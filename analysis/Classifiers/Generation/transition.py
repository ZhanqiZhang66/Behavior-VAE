
#%% 
import csv
import os
import numpy as np
from vame.analysis.community_analysis import read_config, compute_transition_matrices
import matplotlib.pyplot as plt
from utils import load_motif_labels, compute_l0_entropy, effective_num_states, load_tmatrices, save_tmatrices

#%%
def generate_tmatrices(labels, motifSize, split = 1):
    matrices = {}
    frames = len(labels[next(iter(labels))]) // split
    for v in videos:
        matrices[v] = []
        for i in range(split):
            matrix = compute_transition_matrices([v], [labels[v][frames * i: frames * (i + 1)]], motifSize)
            matrices[v].append(matrix[0].tolist())
    return matrices

def generate_tgraphs(path, matrices, motifSize, split = 1):
    for v in videos:
        for i in range(split):
            fig, axes = plt.subplots(1, 1, figsize=(3,3))
            im = axes.imshow(matrices[v][i], cmap='viridis')
            axes.set_xticks(np.arange(motifSize), np.arange(motifSize))
            axes.set_yticks(np.arange(motifSize), np.arange(motifSize))
            plt.colorbar(im, ax=axes)
            plt.grid(None)

            population = "BD"
            if v in hc:
                population = "HC"

            fname = "{}-{}_epoch_{}-{}.png".format(population, v, i, split)
            fname_pdf = "{}-{}_epoch_{}-{}.pdf".format(population, v, i, split)
            fig.savefig(os.path.join(path, fname), transparent=True)
            fig.savefig(os.path.join(path, fname_pdf), transparent=True)

def generate_ens(matrices, split):
    entropy = {}
    for v in videos:
        entropy[v] = []
        for i in range(split):
            entropy[v].append(effective_num_states(matrices[v][i])[1])
    return entropy

def generate_entropy(matrices, split, motifSize):
    entropy = {}
    for v in videos:
        entropy[v] = []
        for i in range(split):
            print(len(matrices[v][i]))
            entropy[v].append(compute_l0_entropy(np.array(matrices[v][i]), motifSize))
    return entropy

def export_data(path, entropy, split):
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator="\n")
        header = ['video']
        for i in range(split):
            header.append('split' + str(i))
        csvwriter.writerow(header)

        for v in videos:
            row = [v]
            row.extend(entropy[v])
            csvwriter.writerow(row)

#%% config
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

hc = ["BC1AASA", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", "BC1HETR", "BC1JECO", "BC1JUPA", "BC1MOKI", "BC1NITA"]


#%%
vamePath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
vameMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix_3\{}_epoch_{}-{}.npy"

hBPMPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
hBPMMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\transition_matrix_3\{}_epoch_{}-{}.npy"

S3DPath = r'C:\Users\kietc\SURF\jack-data\S3D\s3d_labels\s3d_labels_{}.npy'
S3DMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\transition_matrix_3\{}_epoch_{}-{}.npy"

MMActionPath = r'C:\Users\kietc\SURF\jack-data\MMAction\mmaction_labels\mmaction_labels_{}.npy'
MMActionMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\transition_matrix_3\{}_epoch_{}-{}.npy"

#%%
vameLabels = load_motif_labels(vamePath, videos, 27000)
hBPMLabels = load_motif_labels(hBPMPath, videos, 27000)
S3DLabels = load_motif_labels(S3DPath, videos, 27000)
MMActionLabels = load_motif_labels(MMActionPath, videos, 27000)

#%%
vameTMatrices = generate_tmatrices(vameLabels, 10, split = 3)
hBPMTMatrices = generate_tmatrices(hBPMLabels, 10, split = 3)
S3DTMatrices = generate_tmatrices(S3DLabels, 400, split = 3)
MMActionTMatrices = generate_tmatrices(MMActionLabels, 81, split = 3)

#%%
save_tmatrices(vameMatrixPath, videos, vameTMatrices, 3)
save_tmatrices(hBPMMatrixPath, videos, hBPMTMatrices, 3)
save_tmatrices(S3DMatrixPath, videos, S3DTMatrices, 3)
save_tmatrices(MMActionMatrixPath, videos, MMActionTMatrices, 3)


#%%
vameTMatrices = load_tmatrices(vameMatrixPath, videos, split = 3)
hBPMTMatrices = load_tmatrices(hBPMMatrixPath, videos, split = 3)
S3DTMatrices = load_tmatrices(S3DMatrixPath, videos, split = 3)
MMActionTMatrices = load_tmatrices(MMActionMatrixPath, videos, split = 3)


#%%
vameEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
hBPMEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\entropy_3_split.csv'
S3DEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\entropy_3_split.csv'
MMActionEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\entropy_3_split.csv'

#%%
vameEntropy = generate_entropy(vameTMatrices, 3, 9)
hBPMEntropy = generate_entropy(hBPMTMatrices, 3, 10)
S3DEntropy = generate_entropy(S3DTMatrices, 3, 400)
MMActionEntropy = generate_entropy(MMActionTMatrices, 3, 80)

# %%
vameENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\ens_3_split.csv'
hBPMENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\ens_3_split.csv'
S3DENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\ens_3_split.csv'
MMActionENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\ens_3_split.csv'

#%%
vameENS = generate_ens(vameTMatrices, 3)
hBPMENS = generate_ens(hBPMTMatrices, 3)
S3DENS = generate_ens(S3DTMatrices, 3)
MMActionENS = generate_ens(MMActionTMatrices, 3)

# %%
export_data(vameENSPath, vameENS, 3)
export_data(hBPMENSPath, hBPMENS, 3)
export_data(S3DENSPath, S3DENS, 3)
export_data(MMActionENSPath, MMActionENS, 3)

# %%
