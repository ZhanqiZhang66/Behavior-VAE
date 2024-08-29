# Created by zhanq at 7/5/2024
# File:
# Description:
# Scenario:
# Usage
# Created by zhanq at 5/20/2024
# File:
# Description: this need the environment of py39 on victoria-work
# Scenario:
# Usage
from plotting.get_paths import get_my_path

myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']
'''
Colors
'''
b_o_colors = ['#1f77b4', '#ff7f0e']

'''
Load
'''
project_name = 'BD25-HC25-final-May17-2023'
project_path = f'{onedrive_path}\Behavior_VAE_data\{project_name}'
import pandas as pd
import itertools
from multiprocessing import Pool
import numpy as np
from scipy.linalg import sqrtm
import pickle
def wasserstein_distance(m1, C1, m2, C2):
    """
    Calculate the 2-Wasserstein distance between two Gaussian distributions.

    Parameters:
    m1, m2: Mean vectors of the two Gaussian distributions (numpy arrays).
    C1, C2: Covariance matrices of the two Gaussian distributions (numpy arrays).

    Returns:
    W2: The 2-Wasserstein distance.
    """
    # Euclidean distance between the means
    mean_diff = np.linalg.norm(m1 - m2)

    # Principal square roots of the covariance matrices
    # Calculate the trace term
    term = sqrtm(sqrtm(C2) @ C1 @ sqrtm(C2))
    trace_term = np.trace(C1 + C2 - 2 * term)

    # Wasserstein distance squared
    W2_squared = mean_diff ** 2 + trace_term

    return np.sqrt(W2_squared).real

def calculate_representation_shift(args):
    epoch1, motif, df, is_BD = args
    epoch2 = epoch1 + 1
    df_epoch_motif_HC = df[(df['Epoch'] == epoch1) & (df['Motif'] == motif) & (df['is_BD'] == is_BD)]
    df_next_epoch_motif_HC = df[(df['Epoch'] == epoch2) & (df['Motif'] == motif) & (df['is_BD'] == is_BD)]

    print(f"motif{motif} between epoch {epoch1} and epoch {epoch2}")
    distances = []
    for i, in (range(len(df_epoch_motif_HC))):
        latent_vector_HC = df_epoch_motif_HC.iloc[i]['Latent_Vector']
        latent_vector_HC_next_epoch = df_next_epoch_motif_HC.iloc[j]['Latent_Vector']

        print(f"HC-HC  person{i} ")
        if len(latent_vector_HC) <= 1 or len(latent_vector_HC_next_epoch) <= 1:
            representation_shift_distance = np.nan
            print(" skip")
        else:
            m1, C1, m2, C2 = np.mean(latent_vector_HC_next_epoch, axis=0), \
                             np.cov(latent_vector_HC_next_epoch.T), \
                             np.mean(latent_vector_HC, axis=0), \
                             np.cov(latent_vector_HC.T)
            representation_shift_distance = wasserstein_distance(m1, C1, m2, C2)

        distances.append({'FirstEpoch': epoch1, 'Motif': motif, 'is_BD': is_BD, 'Distance': representation_shift_distance})


    return distances

if __name__ == '__main__':
    n_cluster = 10
    with open(f'{project_path}/data/latent_vectors.pkl', 'rb') as f:
        df = pickle.load(f)

    args_list = [(start_epoch, motif, df, is_BD) for start_epoch in range(1, 3) for motif in range(n_cluster) for is_BD in [0, 1]]
    p = Pool(5)
    print("start")
    with p:
        results = p.map(calculate_representation_shift, args_list)
    print("done")
    representation_shift = [item for sublist in results for item in sublist]

    with open(f'{project_path}/data/Representation_shift.pkl', 'wb') as f:
        pickle.dump(distances_BD_HC, f)
   



