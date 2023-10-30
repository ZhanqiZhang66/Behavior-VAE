
#%% 
import csv
import os
import numpy as np
from vame.analysis.community_analysis import read_config, compute_transition_matrices
import matplotlib.pyplot as plt
from utils import loadLabels, compute_entropy

#%% config
path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

hc = ["BC1AASA", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", "BC1HETR", "BC1JECO", "BC1JUPA", "BC1MOKI", "BC1NITA"]

dataframe = {}
n_cluster = 10

matrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix"
#figPath = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\figure\hBPM_transition_matrices"
figPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_figure"

matrices = {}

#%%
def generateTransitionMatrix(labels, split, motifSize):
    matrices = {}
    range = len(labels[0]) // split
    for v in videos:
        matrices[v] = []
        for i in range(split):
            matrix = compute_transition_matrices([v], [labels[v][range * i: range * (i + 1)]], motifSize)
            matrices[v].append(matrix[0])
    return matrices

def exportTransitionMatrix(path, matrices, split):
    for v in videos:
        for i in range(split):
            fname = "{}_epoch_{}-{}.npy".format(v, i, split)
            np.save(os.path.join(path, fname), np.array(matrices[v][i]))

def exportTransitionGraph(path, matrices, split):
    for v in videos:
        for i in range(split):
            fig, axes = plt.subplots(1, 1, figsize=(3,3))
            im = axes.imshow(matrices[v][i], cmap='viridis')
            axes.set_xticks(np.arange(n_cluster), np.arange(n_cluster))
            axes.set_yticks(np.arange(n_cluster), np.arange(n_cluster))
            plt.colorbar(im, ax=axes)
            plt.grid(None)

            population = "BD"
            if v in hc:
                population = "HC"

            fname = "{}-{}_epoch_{}-{}.png".format(population, v, i, split)
            fname_pdf = "{}-{}_epoch_{}-{}.pdf".format(population, v, i, split)
            fig.savefig(os.path.join(path, fname), transparent=True)
            fig.savefig(os.path.join(path, fname_pdf), transparent=True)

