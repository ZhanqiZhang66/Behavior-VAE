# Created by zhanq at 5/20/2024
# File:
# Description:
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

def calculate_HC_distances(args):
    epoch, motif, df = args
    df_epoch_motif_HC = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 0)]

    print(f"motif{motif} epoch {epoch}")
    distances = []
    for i, j in itertools.combinations(range(len(df_epoch_motif_HC)), 2):
        latent_vector_HC = df_epoch_motif_HC.iloc[i]['Latent_Vector']
        latent_vector_HC2 = df_epoch_motif_HC.iloc[j]['Latent_Vector']
        print(f"HC-HC  person{i} vs person {j}")
        if len(latent_vector_HC) <= 1 or len(latent_vector_HC2) <= 1:
            HC_HC_distance = np.nan
            print(" skip")
        else:
            m1, C1, m2, C2 = np.mean(latent_vector_HC2, axis=0), np.cov(latent_vector_HC2.T), np.mean(latent_vector_HC, axis=0), np.cov(latent_vector_HC.T)
            HC_HC_distance = wasserstein_distance(m1, C1, m2, C2)

        distances.append({'Epoch': epoch, 'Motif': motif, 'Distance': HC_HC_distance})


    return distances
def calculate_BD_distances(args):
    epoch, motif, df = args
    df_epoch_motif_BD = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 1)]

    print(f"motif{motif} epoch {epoch}")
    distances = []
    for i, j in itertools.combinations(range(len(df_epoch_motif_BD)), 2):
        latent_vector_BD = df_epoch_motif_BD.iloc[i]['Latent_Vector']
        latent_vector_BD2 = df_epoch_motif_BD.iloc[j]['Latent_Vector']
        print(f"BD-BD  person{i} vs person {j}")
        # Compute distance between the latent vectors
        if len(latent_vector_BD) <= 1  or len(latent_vector_BD2) <= 1 :
            BD_BD_distance = np.nan
            print(" skip")
        else:
            m1, C1, m2, C2 = np.mean(latent_vector_BD, axis=0), np.cov(latent_vector_BD.T), np.mean(latent_vector_BD2,
                                                                                                    axis=0), np.cov(
                latent_vector_BD2.T)
            BD_BD_distance = wasserstein_distance(m1, C1, m2, C2)

        distances.append({'Epoch': epoch, 'Motif': motif, 'Distance': BD_BD_distance})


    return distances
def calculate_BD_HC_distances(args):
    epoch, motif, df = args
    df_epoch_motif_HC = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 0)]
    df_epoch_motif_BD = df[(df['Epoch'] == epoch) & (df['Motif'] == motif) & (df['is_BD'] == 1)]

    print(f"motif{motif} epoch {epoch}")
    distances = []
    for i, j in itertools.combinations(range(len(df_epoch_motif_HC)), 2):
        latent_vector_HC = df_epoch_motif_HC.iloc[i]['Latent_Vector']
        latent_vector_BD = df_epoch_motif_BD.iloc[j]['Latent_Vector']
        print(f"HC-BD  person{i} vs person {j}")
        # Compute distance between the latent vectors
        if len(latent_vector_HC) <= 1 or len(latent_vector_BD) <= 1 :
            BD_HC_distance = np.nan
            print(" skip")
        else:
            m1, C1, m2, C2 = np.mean(latent_vector_HC, axis=0), np.cov(latent_vector_HC.T), np.mean(latent_vector_BD, axis=0), np.cov(latent_vector_BD.T)
            BD_HC_distance = wasserstein_distance(m1, C1, m2, C2)
            print(" computed distance")

        # Add the distance to the distances DataFrame
        distances.append({'Epoch': epoch, 'Motif': motif, 'Distance': BD_HC_distance})

    return distances
if __name__ == '__main__':
    n_cluster = 10
    with open(f'{project_path}/data/latent_vectors.pkl', 'rb') as f:
        df = pickle.load(f)

    args_list = [(epoch, motif, df) for epoch in range(1, 4) for motif in range(n_cluster)]
    p = Pool(5)
    print("start")
    with p:
        results = p.map(calculate_BD_HC_distances, args_list)
        results2 = p.map(calculate_HC_distances, args_list)
        results3 = p.map(calculate_BD_distances, args_list)
    print("done")
    distances_BD_HC = [item for sublist in results for item in sublist]
    distances_HC_HC = [item for sublist in results2 for item in sublist]
    distances_BD_BD = [item for sublist in results3 for item in sublist]


    with open(f'{project_path}/data/Wasserstein_distances_BD_HC.pkl', 'wb') as f:
        pickle.dump(distances_BD_HC, f)
    with open(f'{project_path}/data/Wasserstein_distances_HC_HC.pkl', 'wb') as f:
        pickle.dump(distances_HC_HC, f)
    with open(f'{project_path}/data/Wasserstein_distances_BD_BD.pkl', 'wb') as f:
        pickle.dump(distances_BD_BD, f)



