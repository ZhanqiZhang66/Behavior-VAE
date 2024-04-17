
#%% 
import csv
import os
import numpy as np
from vame.analysis.community_analysis import read_config, compute_transition_matrices
import matplotlib.pyplot as plt
from utils import load_motif_labels, count_zeros, compute_l0_entropy, effective_num_states, load_tmatrices, save_tmatrices

#%%
epoch_duration = 9000 # frames

#%%
#Behavior_VAE_data\BD25-HC25-final-May17-2023\results\BC1AASA\VAME\kmeans-10\latent_vector_BC1AASA.npy
#np.trace(np.cov(epoch_1_latent_vector.T))
def zero_center(vector):
  """Zero centers a vector.
  Args:
    vector: A NumPy array.
  Returns:
    A NumPy array with the same shape as `vector`, but with the mean subtracted
    from each element.
  """
  mean = np.mean(vector)
  return vector - mean
def epoch_motif_volume(epoch_idx, epoch_duration, overall_latent_vector):
    zero_centered_overall_latent_vector = zero_center(overall_latent_vector)

    epoch_motif_latent_vector = zero_centered_overall_latent_vector[epoch_idx * epoch_duration: epoch_idx * epoch_duration + epoch_duration, :]
    partial_latent_volume = np.trace(np.cov(epoch_motif_latent_vector.T))
    overall_latent_volume = np.trace(np.cov(overall_latent_vector.T))
    return partial_latent_volume/overall_latent_volume
def compute_latent_motif_volume(path):
    volume = {}
    for v in videos:
        volume[v]= []
        latent = np.load(path.format(v, v))
        for i in range(3):
            epoch = latent[i * epoch_duration: i * epoch_duration + epoch_duration, :]
            volume[v].append(np.trace(np.cov(epoch.T)))
    return volume

# %%
def save_motif_volume(path, volume):
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator="\n")
        header = ['video', 'split0', 'split1', 'split2']
        csvwriter.writerow(header)

        for v in videos:
            row = [v]
            for i in range(3):
                row.append(volume[v][i])
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
path = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\latent_vector_{}.npy"

#%%
volume = compute_latent_motif_volume(path)
# %%
volumePath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\volume.csv"
save_motif_volume(volumePath, volume)
# %%
