
#%% 
import csv
import os
import numpy as np
from vame.analysis.community_analysis import read_config, compute_transition_matrices
import matplotlib.pyplot as plt
from utils import loadLabels, compute_entropy

#%%
def generateTransitionMatrix(labels, motifSize, split = 1):
    matrices = {}
    frames = len(labels[next(iter(labels))]) // split
    for v in videos:
        matrices[v] = []
        for i in range(split):
            matrix = compute_transition_matrices([v], [labels[v][frames * i: frames * (i + 1)]], motifSize)
            matrices[v].append(matrix[0])
    return matrices

def exportTransitionMatrix(path, matrices, split = 1):
    for v in videos:
        for i in range(split):
            fname = "{}_epoch_{}-{}.npy".format(v, i, split)
            np.save(os.path.join(path, fname), np.array(matrices[v][i]))

def exportTransitionGraph(path, matrices, motifSize, split = 1):
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

#%% config
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

hc = ["BC1AASA", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", "BC1HETR", "BC1JECO", "BC1JUPA", "BC1MOKI", "BC1NITA"]

#%%
dataframe = {}
n_cluster = 10

path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
matrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix"
#figPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\hBPM_transition_matrices"
figPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_figure"

matrices = {}


#%%
vamePath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
vameMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix_3"
vameFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_figure_3"

hBPMPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
hBPMMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\transition_matrix_3"
hBPMFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\transition_figure_3"

S3DPath = r'C:\Users\kietc\SURF\jack-data\S3D\s3d_labels\s3d_labels_{}.npy'
S3DMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\transition_matrix_3"
S3DFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\transition_figure_3"

MMActionPath = r'C:\Users\kietc\SURF\jack-data\MMAction\mmaction_labels\mmaction_labels_{}.npy'
MMActionMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\transition_matrix_3"
MMActionFigPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\transition_figure_3"

#%%
vameLabels = loadLabels(vamePath, videos, 27000)
hBPMLabels = loadLabels(hBPMPath, videos, 27000)
S3DLabels = loadLabels(S3DPath, videos, 27000)
MMActionLabels = loadLabels(MMActionPath, videos, 27000)

#%%
vameTransitionMatrix = generateTransitionMatrix(vameLabels, 10, split = 3)
hBPMTransitionMatrix = generateTransitionMatrix(hBPMLabels, 10, split = 3)
S3DTransitionMatrix = generateTransitionMatrix(S3DLabels, 400, split = 3)
MMActionTransitionMatrix = generateTransitionMatrix(MMActionLabels, 81, split = 3)

#%%
exportTransitionMatrix(vameMatrixPath, vameTransitionMatrix)
exportTransitionMatrix(hBPMMatrixPath, hBPMTransitionMatrix)
exportTransitionMatrix(S3DMatrixPath, S3DTransitionMatrix)
exportTransitionMatrix(MMActionMatrixPath, MMActionTransitionMatrix)

#%%
exportTransitionGraph(vameFigPath, vameTransitionMatrix, 10)
exportTransitionGraph(hBPMFigPath, hBPMTransitionMatrix, 11)
exportTransitionGraph(S3DFigPath, S3DTransitionMatrix, 400)
exportTransitionGraph(MMActionFigPath, MMActionTransitionMatrix, 81)

# %%
def effective_num_states(transtion_m):
    effective_num_every_state = []
    for row in transtion_m:
        sum_p_ij = np.sum(np.square(row))
        if sum_p_ij == 0:
            effective_num_every_state.append(0)
        else:
            effective_num_every_state.append(1/sum_p_ij)
    effective_num_avg = np.mean(effective_num_every_state)
    return effective_num_every_state, effective_num_avg

def generateEntropy(matrices, split):
    entropy = {}
    for v in videos:
        entropy[v] = []
        for i in range(split):
            entropy[v].append(effective_num_states(matrices[v][i])[1])
    return entropy

def exportEntropy(path, entropy, split):
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

#%%
vameEntropy = generateEntropy(vameTransitionMatrix, 3)
hBPMEntropy = generateEntropy(hBPMTransitionMatrix, 3)
S3DEntropy = generateEntropy(S3DTransitionMatrix, 3)
MMActionEntropy = generateEntropy(MMActionTransitionMatrix, 3)

#%%
vameEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
hBPMEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\entropy_3_spli.csv'
S3DEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\entropy_3_spli.csv'
MMActionEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\entropy_3_spli.csv'


# %%
exportEntropy(vameEntropyPath, vameEntropy, 3)
exportEntropy(hBPMEntropyPath, hBPMEntropy, 3)
exportEntropy(S3DEntropyPath, S3DEntropy, 3)
exportEntropy(MMActionEntropyPath, MMActionEntropy, 3)

# %%
def count_zeros(transition_m):
    transition = transition_m.copy()
    zero_rows = np.all(transition == 0, axis=1)
    zero_rows_i =  np.where(zero_rows == True)
    zero_cols = np.all(transition == 0, axis=0)
    return len(zero_rows_i[0]),  np.count_nonzero(transition == 1), np.count_nonzero(transition == 0)
