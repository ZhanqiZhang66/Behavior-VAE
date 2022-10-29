import os
import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, silhouette_samples
from vame.util.auxiliary import read_config
from vame.model.rnn_model import RNN_VAE, RNN_VAE_LEGACY
#%%
def goodness_of_fit(cfg, files, latent_vector_files, cluster):
    # https://github.com/rahulkhandelwal396/kmeans-evaluation/blob/main/kmeans%20evaluation.ipynb
    # Getting the sum of squared distances for different values of k
    random_state = cfg['random_state_kmeans']
    n_init = cfg['n_init_kmeans']
    n_cluster = cfg['n_cluster']
    latent_vector_cat = np.concatenate(latent_vector_files, axis=0)
    sum_of_squared_distances = []
    silhouette = []
    n = range(1, cluster)
    for i in n:
        km = KMeans(init='k-means++', n_clusters=i, random_state=random_state, n_init=n_init)
        km1 = km.fit(latent_vector_cat)
        preds = km.fit_predict(latent_vector_cat)
        score = silhouette_score(latent_vector_cat, preds)
        silhouette.append(score)
        sum_of_squared_distances.append(km1.inertia_)

    # calinski_harabasz
    model = KMeans()
    visualizer = KElbowVisualizer(
        model, k=(2, cluster), metric='calinski_harabasz', timings=False
    )
    visualizer.fit(latent_vector_cat)  # Fit the data to the visualizer
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    visualizer.show()  # Finalize and render the figure

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(n, sum_of_squared_distances, 'bx-')
    ax.set_xlabel('k')
    ax.set_ylabel('sum_of_squared_distances')
    ax.set_title('Elbow Method For Optimal k')
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    n_clusters = [x for x in range(2, cluster)]
    df_silhouette = pd.DataFrame({'n_clusters': n_clusters, 'silhouette_score': silhouette})
    sns.lineplot(data=df_silhouette, x="n_clusters", y="silhouette_score")
    fig.show()





    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

    return sum_of_squared_distances
#%%
project_name = 'BD20-Jun5-2022'
config = 'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}\config.yaml'.format(project_name)
from vame.analysis.community_analysis import  read_config
cfg = read_config(config)
legacy = cfg['legacy']
model_name = cfg['model_name']
n_cluster = cfg['n_cluster']
files = []
for file in cfg['video_sets']:
    files.append(file)
latent_vectors = []
for file in files:
    path_to_latent_vector = os.path.join(cfg['project_path'], "results", file, model_name,'kmeans-' + str(n_cluster), "")
    latent_vector = np.load(os.path.join(path_to_latent_vector, 'latent_vector_' + file + '.npy'))
    latent_vectors.append(latent_vector)
#%%
n_cluster = 3
random_state = cfg['random_state_kmeans']
n_init = cfg['n_init_kmeans']

latent_vector_cat = np.concatenate(latent_vectors, axis=0)
sum_of_squared_distances = []
silhouette = []
n = range(2, n_cluster)
for i in n:
    km = KMeans(init='k-means++', n_clusters=i, random_state=random_state, n_init=n_init)
    km1 = km.fit(latent_vector_cat)
    preds = km.fit_predict(latent_vector_cat)
    score = silhouette_score(latent_vector_cat, preds)
    silhouette.append(score)
    sum_of_squared_distances.append(km1.inertia_)
#%%
# calinski_harabasz
model = KMeans()
visualizer = KElbowVisualizer(
    model, k=(2, n_cluster), metric='calinski_harabasz', timings=False
)
visualizer.fit(latent_vector_cat)  # Fit the data to the visualizer
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
visualizer.show()  # Finalize and render the figure

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(n, sum_of_squared_distances, 'bx-')
ax.set_xlabel('k')
ax.set_ylabel('sum_of_squared_distances')
ax.set_title('Elbow Method For Optimal k')
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
n_clusters = [x for x in range(2, n_cluster)]
df_silhouette = pd.DataFrame({'n_clusters': n_clusters, 'silhouette_score': silhouette})
sns.lineplot(data=df_silhouette, x="n_clusters", y="silhouette_score")
fig.show()