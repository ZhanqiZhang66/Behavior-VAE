
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

control_videos = ["BC1AASA", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", "BC1HETR", "BC1JECO", "BC1JUPA", "BC1MOKI", "BC1NITA"]
BD_videos = [v for v in videos if v not in control_videos]
#%%
path = r"C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\latent_vector_{}.npy"

#%%
volume = compute_latent_motif_volume(path)
# %%
volumePath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\volume.csv"
save_motif_volume(volumePath, volume)
# %%
#%% Reading VAME data
jack_temp = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\jack_temp\{}'
Epoch1_latent_vector = np.load(jack_temp.format('Epoch1_latent_vector.npy'), allow_pickle=True).tolist()
Epoch2_latent_vector = np.load(jack_temp.format('Epoch2_latent_vector.npy'), allow_pickle=True).tolist()
Epoch3_latent_vector = np.load(jack_temp.format('Epoch3_latent_vector.npy'), allow_pickle=True).tolist()
Epoch1_labels = np.load(jack_temp.format('Epoch1_labels.npy'), allow_pickle=True).tolist()
Epoch2_labels = np.load(jack_temp.format('Epoch2_labels.npy'), allow_pickle=True).tolist()
Epoch3_labels = np.load(jack_temp.format('Epoch3_labels.npy'), allow_pickle=True).tolist()

# %% Compute centroids of motif-volume per person, and per population
n_subject_in_population = 25
n_cluster = 10
zdim = 10

person_centroids = np.empty(
    (3, 2, n_subject_in_population, n_cluster, zdim))  # epoch x pop x subject x 10 motifs x zdim
person_volumes = np.empty((3, 2, n_subject_in_population, n_cluster))
population_centroids = np.empty((3, 2, n_cluster, zdim))
population_volumes = np.empty((3, 2, n_cluster))
for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        latent_vec_this_epoch_this_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        label_this_epoch_this_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for sub in range(n_subject_in_population):
            latent_vec_this_epoch_this_pop_this_person = latent_vec_this_epoch_this_pop[9000 * sub: 9000 * (sub + 1)]
            label_this_epoch_this_pop_this_person = label_this_epoch_this_pop[9000 * sub: 9000 * (sub + 1)]
            for g in range(n_cluster):
                idx_g = np.where(label_this_epoch_this_pop_this_person == g)[0]
                latent_vec_this_epoch_this_pop_this_person_this_motif = latent_vec_this_epoch_this_pop_this_person[
                    idx_g]
                if len(latent_vec_this_epoch_this_pop_this_person_this_motif):
                    person_centroid = np.nanmean(latent_vec_this_epoch_this_pop_this_person_this_motif, axis=0)
                    person_volume = np.trace(np.cov(latent_vec_this_epoch_this_pop_this_person_this_motif.T))
                else:
                    person_centroid = np.full([zdim, ], np.nan)
                    person_volume = 0
                person_centroids[epoch - 1, j, sub, g, :] = person_centroid
                person_volumes[epoch - 1, j, sub, g] = person_volume


# Note: there is a small offset
print(np.nanmean(person_centroids[1, 0, :, 0, :], axis=0))
print(population_centroids[1, 0, 0, :])
# %%
volume = {}
for epoch in range(3):
    for j, videos in enumerate([control_videos, BD_videos]):
        for sub in range(n_subject_in_population):
            if j == 0:
                v = control_videos[sub]
            else:
                v = BD_videos[sub]
            if v not in volume:
              volume[v] = []
            volume[v].append(person_volumes[epoch-1, j, sub])
              

# %%
# %%
def save_volume_per_motif(path, volume):
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator="\n")
        header = ['video']
        for i in range(3):
            for m in range(10):
                header.append('{}motif{}'.format(i, m))
                
        csvwriter.writerow(header)

        for v in videos:
            row = [v]
            for i in range(3):
                row.extend(volume[v][i])
            csvwriter.writerow(row)

# %%
volumePerMotfPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\volume_per_motif.csv"
save_volume_per_motif(volumePerMotfPath, volume)
# %%
