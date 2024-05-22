import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import scipy
import seaborn as sns
from scipy import stats

#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag

from data.load_data import load_pt_data
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import distance
import itertools
from scipy.stats import wasserstein_distance_nd
#%%
'''
Paths
'''
if os.environ['COMPUTERNAME'] == 'VICTORIA-WORK':
    onedrive_path = r'C:\Users\zhanq\OneDrive - UC San Diego'
    github_path = r'C:\Users\zhanq\OneDrive - UC San Diego\GitHub'
elif os.environ['COMPUTERNAME'] == 'VICTORIA-PC':
    github_path = r'D:\OneDrive - UC San Diego\GitHub'
    onedrive_path = r'D:\OneDrive - UC San Diego'

else:
    github_path = r'D:\OneDrive - UC San Diego\GitHub'


'''
Colors
'''
b_o_colors = ['#1f77b4', '#ff7f0e']

'''
Load
'''
project_name = 'BD25-HC25-final-May17-2023'
project_path = f'{onedrive_path}\Behavior_VAE_data\{project_name}'

dlc_path = os.path.join(project_path,"videos","\pose_estimation") #dlc_path = 'D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/{}'.format(project_name)
n_cluster = 10
zdim = 10
model_name = 'VAME'
five_min_frame_no = int(5 * 60 * 30)
data, YMRS, HAM_D, gender, start_frame, condition, isBD = load_pt_data(video_information_pth=
                                                                       r'{}\Behavior-VAE\data\video-information.csv'.format(github_path))
control_videos = [k for k, v in isBD.items() if v[0] == 'healthy']
BD_videos = [k for k, v in isBD.items() if v[0] == 'Euthymic']
score_bahavior_names =["sit", "sit_obj", "stand", "stand-obj", "walk", "walk_obj", "lie", "lie_obj", "interact", "wear", "exercise"]
n_subject_in_population = len(control_videos)
cmap = plt.get_cmap('tab20')
titles = ["CP", "BD"]
pca = PCA(n_components=3)
#%%
'''
Define Variables of interests
'''
titles = ["CP", "BD"]
N = [0, 0]


Motif_usages = [[], []]
Motif_usage_pct = [[], []]
motif_usage_cat = [[], []]
Latent_vectors = [[], []]
Latent_centroids = []  # this is the mean of the latent, on d-dimension
Latent_len = [[],[]]
Latent_len_epoch = [[[],[]], [[],[]], [[],[]]]
Labels = [[], []]
TM = [[], []]
population_TM = [[], []]
Cluster_center = []
transition_matrices = []

Latent_vectors_per_state = [[],[]] # mean and variance of state 0,1,...9 of every person's latent vector
State_vectors_per_latent_d = [[],[]] # mean and variance of latent d = 0,1,...9 of every person's state
volume_per_person = [[],[]] # trace of cov(latent vector of this person)
Stats_per_latent_per_state = [[],[]]


Epoch1_labels = [[], []]
Epoch1_latent_vector = [[], []]
Epoch1_volume_per_person = [[],[]] # trace of cov(latent vector of this person)


Epoch2_labels = [[], []]
Epoch2_latent_vector = [[], []]
Epoch2_volume_per_person = [[],[]] # trace of cov(latent vector of this person)

Epoch3_labels = [[], []]
Epoch3_latent_vector = [[], []]
Epoch3_volume_per_person = [[],[]] # trace of cov(latent vector of this person)
#%%
'''
Step 1: Mean-centered the population-latent vectors for computing
'''

# Get the latent vector of each video
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Loading {} data...".format(v))

        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))

        folder = os.path.join(project_path, "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # T x d = {10, 30}
        latent_vector = latent_vector[: five_min_frame_no*3]
        label = label[: five_min_frame_no*3]
        if i == 0:
            l = label
            latent = latent_vector
        else:
            latent = np.concatenate([latent, latent_vector])
            l = np.concatenate([l,label])
    Latent_len[j].append(len(latent_vector))
    Latent_vectors[j] = latent
    Labels[j] = l

# compute the mean of latent population
population_latent_vector = np.vstack(Latent_vectors)
population_latent_vector_centroid = np.mean(population_latent_vector, axis=0)
Latent_centroids = population_latent_vector_centroid

#%%
'''
Step 2: compute overall latent motif-volume, and epoch latent motif-volume
'''
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Mean-Centering {} data...".format(v))
        label = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name, v,n_cluster,n_cluster,v))
        cluster_center = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        motif_usage = np.load(r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name, v,n_cluster, v))
        folder = os.path.join(project_path, "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # T x d = {10, 30}
        Latent_len[j].append(len(latent_vector))

        # variance of each person
        K = np.cov(latent_vector.T)  # covariance matrix of the 10-D latent vector
        volume = np.trace(K)
        volume_per_person[j].append(volume)

        door_close_time = start_frame[v]
        start_time = door_close_time

        offset = 0 #int(start_time - door_close_time)

        epoch_1_label = label[:five_min_frame_no + offset]
        epoch_2_label = label[five_min_frame_no + offset: five_min_frame_no * 2 + offset]
        epoch_3_label = label[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]
        Latent_len_epoch[0][j].append(len(epoch_1_label))
        Latent_len_epoch[1][j].append(len(epoch_2_label))
        Latent_len_epoch[2][j].append(len(epoch_3_label))

   

        # mean-centered epoch latent vector
        epoch_1_latent_vector = latent_vector[:five_min_frame_no + offset] - Latent_centroids
        epoch_2_latent_vector = latent_vector[five_min_frame_no + offset: five_min_frame_no * 2 + offset] - Latent_centroids
        epoch_3_latent_vector = latent_vector[five_min_frame_no * 2 + offset: five_min_frame_no * 3 + offset]- Latent_centroids


        Epoch1_volume_per_person[j].append(np.trace(np.cov(epoch_1_latent_vector.T)))
        Epoch2_volume_per_person[j].append(np.trace(np.cov(epoch_2_latent_vector.T)))
        Epoch3_volume_per_person[j].append(np.trace(np.cov(epoch_3_latent_vector.T)))


        Epoch1_labels[j].append(epoch_1_label)
        Epoch1_latent_vector[j].append(epoch_1_latent_vector)

        Epoch2_labels[j].append(epoch_2_label)
        Epoch2_latent_vector[j].append(epoch_2_latent_vector)

        Epoch3_labels[j].append(epoch_3_label)
        Epoch3_latent_vector[j].append(epoch_3_latent_vector)

# Initialize an empty DataFrame

# Initialize an empty list to store dictionaries
data_list = []

for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        latent_vec_this_epoch_this_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        label_this_epoch_this_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for g in range(n_cluster):
            for sub_i in range(n_subject_in_population):
                latent_vec_this_epoch_this_pop_this_person = latent_vec_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                label_this_epoch_this_pop_this_person = label_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                idx_g = np.where(label_this_epoch_this_pop_this_person == g)[0]
                latent_vec_this_epoch_this_pop_this_person_this_motif = latent_vec_this_epoch_this_pop_this_person[idx_g]
                # Convert latent vector array to string
                # Create a dictionary to store the data
                data = {
                    'Subject': sub_i,
                    'is_BD': j,
                    'Epoch': epoch,
                    'Motif': g,
                    'Latent_Vector': latent_vec_this_epoch_this_pop_this_person_this_motif
                }
                # Append the dictionary to the list
                data_list.append(data)

# Create DataFrame from the list of dictionaries
df = pd.DataFrame(data_list)

distances_HC_HC = []

for epoch in range(1, 4):
    for motif in range(n_cluster):
        # Select the rows corresponding to J == 0 and J == 1 for the current epoch and motif
        df_epoch_motif_HC = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 0)]


        # Iterate over each pair of latent vectors and compute the distance
        print(f"motif{motif} epoch {epoch}")
        for i, j in itertools.combinations(range(len(df_epoch_motif_HC)), 2):
            latent_vector_HC = df_epoch_motif_HC.iloc[i]['Latent_Vector']
            latent_vector_HC2 = df_epoch_motif_HC.iloc[j]['Latent_Vector']

            # Compute distance between the latent vectors
            if len(latent_vector_HC) == 0 or len(latent_vector_HC2) == 0:
                HC_HC_distance = -1
                print(" 0")
            else:
                HC_HC_distance = wasserstein_distance_nd(latent_vector_HC2, latent_vector_HC)

            # Add the distance to the distances DataFrame
            distances_HC_HC.append({'Epoch': epoch, 'Motif': motif, 'Distance': HC_HC_distance})
            print(f"HC-HC  person{i} vs person {j}")


distances_HC_HC_df = pd.DataFrame(distances_HC_HC)
distances_HC_HC_df.to_csv(f'{project_path}/data/Wasserstein_distances_HC_HC_df.csv', index=False)
