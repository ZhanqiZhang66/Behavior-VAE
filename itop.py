# Created by Victoria Zhang at 4/11/2022
# File: itop.py
# Description:  read ITOP dataset
# Scenario:
#Usage:
import h5py
import numpy as np
import os, sys
from os.path import join
import numpy as np
import vame
import pandas as pd
from vame.analysis.community_analysis import  read_config
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\VAME")   #Victoria local
sys.path.append(r"D:\OneDrive - UC San Diego\GitHub\markov-chain")
#%% 1.0 video files & directories
video_dir = r'G:\hBPM_BD_Control_Iter2\hBPM_BD_Control_Iter2'
videos = []
for filename in os.listdir(video_dir):
    if filename.endswith('.mpg'):
        fpath = os.path.join(video_dir, filename)
        videos.append(fpath)
#%% 1.1 create new config
config = vame.init_new_project(project='BD20',
                               videos=videos,
                               working_directory='D:/OneDrive - UC San Diego/GitHub/hBPMskeleton/',
                               videotype='.mpg')
#%% OR 1.2 load pre-defined config
project_name = 'BD20-Jun5-2022'
config = 'D:\OneDrive - UC San Diego\Bahavior_VAE_data\{}\config.yaml'.format(project_name)

#%% Edit config file
print("edit n_cluster, zdims, num_features=(#dlc labels x2), kmeans_loss=(n_cluster)")

#%% 1.5 load start frame (frame that person completely entered the room)
start_frame = pd.read_csv('G:\start_frame.csv')
start_frame = start_frame.set_index('video_name').T.to_dict('list')
#%% 2. egocentric alignment
vame.egocentric_alignment(config,
                          start_frame=start_frame,
                          pose_ref_index=[5,12],
                          crop_size=(300,300),
                          use_video=False,
                          video_format='.mpg',
                          check_video=False)
#%% 3. create train set
vame.create_trainset(config)
#Lenght of train data: 588815
#Lenght of test data: 65423 (BC1MOKI, last 10794 frames, plus BC1OKBA BC1NITA)
#%% 3.1 Find test dataset frame correspondance
from vame.analysis.community_analysis import  read_config
cfg = read_config(config)
files = []
for file in cfg['video_sets']:
    files.append(file)
for file in files:
    path_to_file = os.path.join(cfg['project_path'], "data", file, file + '-PE-seq.npy')
    X = np.load(path_to_file)
    X_len = len(X.T)
    print(file, ' ', X_len)
#%% 4. train model [take a while]
vame.train_model(config)
#%% 5. evaluate model
vame.evaluate_model(config)
#%% 6. cluster motifs
vame.pose_segmentation(config)

vame.motif_videos(config, start_frame, videoType='.mpg')
#%% 6.1 create tree and umap
vame.community(config, show_umap=True, cut_tree=None)

#%%
from vame.analysis.community_analysis import get_labels, compute_transition_matrices
model_name = cfg['model_name']
n_cluster = 10
files = ['BC1ANGA', 'BC1LOKE']
labels = get_labels(cfg, files, model_name, n_cluster)
transition_matrices = compute_transition_matrices(files, labels, n_cluster)
trees = []
communities_all = []
for idx, file in enumerate(files):
    path_to_file = os.path.join(cfg['project_path'], "results", file, model_name, 'kmeans-' + str(n_cluster), "")
    if not os.path.exists(os.path.join(path_to_file, "community")):
        os.mkdir(os.path.join(path_to_file, "community"))

    np.save(os.path.join(path_to_file, "community", "transition_matrix_" + file + '.npy'), transition_matrices[idx])
    print("Saving trasition matrix for {}".format(file))












#%%
from vame.analysis.community_analysis import get_labels, compute_transition_matrices, graph_to_tree
cfg = read_config(config)
n_cluster = 10
model_name = cfg['model_name']

labels = get_labels(cfg, files, model_name, n_cluster)
transition_matrices = compute_transition_matrices(files, labels, n_cluster)
trees = []
communities_all = []
for idx, file in enumerate(files):
    print("Generate tree for {}".format(file))
    path_to_file = os.path.join(cfg['project_path'], "results", file, model_name, 'kmeans-' + str(n_cluster), "")
    k_labels, usage = np.unique(labels[idx], return_counts=True)
    T = graph_to_tree(usage, k_labels, transition_matrices[idx], n_cluster, merge_sel=1)
    trees.append(T)
    print("Saving tree for {}".format(file))
    np.save(os.path.join(path_to_file, "community", "tree_" + n_cluster + file + '.npy'), T)
#%%  7. visualization
# vame.motif_videos(config, videoType='.mpg')
# vame.visualization(config, label="motif")#options: label: None, "motif", "community"
# #%% 8. make gif
# vame.gif(config, pose_ref_index=[5, 6], subtract_background=False, start=0, length=500, max_lag=30, label='community', file_format='.mpg', crop_size=(1080,1920))
#
# #%% visualize umap
#













#%% Summerize latent information
import umap
import numpy as np
import matplotlib.pyplot as plt
import os
from vame.analysis.community_analysis import  read_config#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
cfg = read_config(config)
n_cluster = 30
model_name = 'VAME'
control_videos = ['BC1ANGA','BC1ANHE','BC1AASA','BC1ALKA','BC1ALPA','BC1ALRO','BC1ANBU','BC1ANWI','BC1ASKA','BC1ATKU','BC1MOKI','BC1NITA']
BD_videos      = ['BC1LOKE','BC1MAMA','BC1ADPI','BC1CISI','BC1DOBO','BC1JUST','BC1KEMA','BC1LABO','BC1LACA','BC1BRBU','BC1MISE','BC1OKBA']
N = [0,0]
Motif_usages = [[],[]]
motif_usage_cat = [[],[]]
Latent_vectors = [[],[]]
Labels = [[], []]
TM = [[], []]
Cluster_center = []
transition_matrices = []
#%%
titles = ["CP", "BD"]
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Loading {} data...".format(v))
        label = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(project_name, v,n_cluster,n_cluster,v))
        transition_m = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(project_name, v,n_cluster, v))
        cluster_center = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(project_name, v,n_cluster, v))
        motif_usage = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(project_name, v,n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy')) # L x 30
        transition_matrices.append(transition_m)
        # plt.figure()
        # im = plt.imshow(transition_m, vmin=0, vmax=np.max(transition_m))
        # plt.title("{}-{} Transition matrix".format(titles[j], v))
        # plt.xticks(np.arange(0, np.shape(transition_m)[0], 1))
        # plt.yticks(np.arange(0, np.shape(transition_m)[0], 1))
        # plt.colorbar(im)
        # plt.tight_layout()
        # plt.box('off')
        # plt.savefig(os.path.join(cfg['project_path'], "Transition_" + str(n_cluster) + v + '.png'))
        # plt.show()
        # plt.close()

        if i == 0:
            l = label
            latent = latent_vector
            m = motif_usage
            tm = transition_m
            Cluster_center.append(cluster_center)
        else:
            latent = np.concatenate([latent, latent_vector])
            l = np.concatenate([l,label])
            m += motif_usage
            tm += transition_m
            Cluster_center.append(cluster_center)
        motif_usage_cat[j].append(motif_usage/ np.sum(motif_usage))
        # num_points = cfg['num_points']
        # if num_points > latent_vector.shape[0]:
        num_points = latent_vector.shape[0]
        n += num_points
    N[j] = n
    Motif_usages[j] = m
    Latent_vectors[j] = latent
    Labels[j] = l
    TM[j] = tm/12

#%% between motif paired t test
motif_usage_cat = np.asarray(motif_usage_cat)
from scipy import stats
for i in range(30):
    CP = motif_usage_cat[0,:,i].reshape(-1,1)
    BD = motif_usage_cat[1,:,i].reshape(-1,1)
    s = stats.ttest_rel(CP, BD)
    print("motif {}:{}, {}".format(i,s.statistic, s.pvalue))
#%%
from vame.analysis.community_analysis import  read_config#, get_labels, compute_transition_matrices, get_community_labels, create_community_bag
cfg = read_config(config)
import matplotlib
import seaborn as sns
import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
path_to_file = cfg['project_path']
filenames = ['BC1ANGA','BC1JUST']
fig, axs = plt.subplots(2, 1, figsize=(8,4))
for j, filename in enumerate(filenames):
    data = pd.read_csv(os.path.join(path_to_file, 'videos', 'pose_estimation', filename + '.csv'), skiprows=2)
    label = np.load(
        r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
            project_name, filename, n_cluster, n_cluster, filename))
    data_mat = pd.DataFrame.to_numpy(data)
    f_start_frame = start_frame[filename][0]
    data_mat = data_mat[f_start_frame:, 1:]

    # get the coordinates for alignment from data table
    pose_list = []

    for i in range(int(data_mat.shape[1] / 3)):
        pose_list.append(data_mat[:, i * 3:(i + 1) * 3])


    time = np.linspace(1,len(pose_list[0]),num=len(pose_list[0]))
    x = pose_list[10][:,0]
    y = pose_list[10][:,1]

    cmap = sns.color_palette(cc.glasbey, n_colors=len(df['color'].unique()))
    fig, axs = plt.subplots(figsize=(8, 4))
    df = pd.DataFrame(list(zip(time, label)),
                   columns =['x', 'color'])
    for k, c in enumerate(df['color'].unique()):
        bounds = df[['x', 'color']].groupby('color').agg(['min', 'max']).loc[c]
        axs.axvspan(bounds.min(), bounds.max()+1, alpha=0.1, color=cmap[k])
    axs.scatter(time, x, s=2, label='x')
    axs.scatter(time, y, s=2, label='y')
    axs.legend(loc="upper right")
    axs.set_xlabel('frames')
    axs.set_ylabel('coordinates')
    axs.set_title(filename)
    plt.tight_layout
    plt.show()
    plt.savefig(os.path.join(cfg['project_path'], "time_" + str(n_cluster) + filename+'.png'))
plt.close()
#%% Plotting markov chain
import networkx as nx
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from hierarchy_pos import hierarchy_pos, topo_pos
from grandalf.layouts import SugiyamaLayout
from networkx.drawing.nx_agraph import graphviz_layout
SCCs = []

titles = ["CP", "BD"]

videos = control_videos + BD_videos
for i in range(len(videos)):
    if i < 12:
        t = 'CP'
    else:
        t = 'BD'
    v = videos[i]
    tm = np.load(r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(project_name, v,n_cluster, v))
    # tm[tm<0.1]=0
    G = nx.DiGraph(tm)

    weight = []
    for (nodefrom, nodeto) in G.edges:
        w = tm[nodefrom, nodeto]
        weight.append(w)
    LWidths = (weight/ max(weight))*5
    nodelist = G.nodes()
    edgeList = G.edges()

    # compute index for the subplot, and set this subplot as current

    # plt.sca(axs[j])
    #pos = graphviz_layout(G, prog='dot')
    # fig, axs = plt.subplots(1, 2)
    # plt.sca(axs[0])
    plt.figure()
    pos = nx.circular_layout(G)
    #pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')
    # nx.draw(G, pos)

    # pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=nodelist,
                           node_size=500,
                           node_color=range(len(nodelist)),
                           alpha=0.8)
    nx.draw_networkx_edges(G, pos,
                           edgelist=edgeList,
                           width=LWidths,
                           edge_color='lightblue',
                           alpha=1,
                           arrows=True,
                           arrowsize=20)
    nx.draw_networkx_labels(G, pos=pos,
                            labels=dict(zip(nodelist, nodelist)),
                            font_color='white')
    plt.title("{}-{}".format(t, v))
    plt.tight_layout()
    plt.box('off')
    plt.savefig(os.path.join(r'D:\OneDrive - UC San Diego\Bahavior_VAE_data\BD20-Jun5-2022\figure\graphs', "net_" + str(n_cluster) + v + '.png'))


    scc = nx.kosaraju_strongly_connected_components(G)
    DAG = nx.condensation(G, scc=scc)
    SCCs.append(list(nx.kosaraju_strongly_connected_components(G)))
    pos = nx.drawing.nx_pydot.pydot_layout(DAG, prog='dot')
    #nx.draw(DAG, pos=pos)
    nodelist = DAG.nodes()
    edgeList = DAG.edges()

    # plt.sca(axs[1])
    # weight = []
    # for (nodefrom, nodeto) in G.edges:
    #     w = tm[nodefrom, nodeto]
    #     weight.append(w)
    # if weight != []:
    #     LWidths = (weight / max(weight)) * 5
    #     nx.draw_networkx_edges(DAG, pos,
    #                            edgelist=edgeList,
    #                            width=LWidths,
    #                            edge_color='lightblue',
    #                            alpha=1,
    #                            arrows=True,
    #                            arrowsize=20)
    # else:
    #     LWidths = 0
    #
    # # # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(DAG, pos,
    #                        nodelist=nodelist,
    #                        node_size=500,
    #                        node_color=range(len(nodelist)),
    #                        alpha=0.8)
    # nx.draw_networkx_labels(DAG, pos=pos,
    #                         labels=dict(zip(nodelist, nodelist)),
    #                         font_color='white')
    # plt.title("DAG {}-{}".format(titles[j], v))
    # plt.tight_layout()
    # plt.box('off')
    # plt.savefig(os.path.join(cfg['project_path'], "tree_" + str(n_cluster) + v + '.png'))
    # plt.show()
#%% Plot community network
import networkx as nx
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from hierarchy_pos import hierarchy_pos, topo_pos
from grandalf.layouts import SugiyamaLayout
from networkx.drawing.nx_agraph import graphviz_layout
from netgraph import Graph
import seaborn as sns
import colorcet as cc
SCCs = []

titles = ["CP", "BD"]
community_labels = []
videos = control_videos + BD_videos
for i in range(len(videos)):
    if i < 12:
        t = 'CP'
    else:
        t = 'BD'
    v = videos[i]
    f_start_frame = start_frame[v][0]
    label = np.load(
        r'D:\OneDrive - UC San Diego\GitHub\hBPMskeleton\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(
            project_name, v, n_cluster, n_cluster, v))
    unique_labels = np.unique(label)
    community_label = np.zeros_like(label)


    tm = transition_matrices[i]
    # tm[tm<0.1]=0
    G = nx.DiGraph(tm)
    #G.remove_nodes_from(list(nx.isolates(G)))

    weight = []
    for node in list(G.nodes):
        if node not in unique_labels:
            G.remove_node(node)
    for (nodefrom, nodeto) in G.edges:
        w = tm[nodefrom, nodeto]
        weight.append(w)
    if weight != []:
        LWidths = (weight/ max(weight))*5
    else:
        LWidths = 1
    nodelist = G.nodes()
    edgeList = G.edges()

    scc = nx.kosaraju_strongly_connected_components(G)
    # DAG = nx.condensation(G, scc=scc)
    SCCs.append(list(nx.kosaraju_strongly_connected_components(G)))

    node_to_community = dict()
    community_to_color = dict()


    communities = list(nx.kosaraju_strongly_connected_components(G))
    community_colors = sns.color_palette(cc.glasbey, n_colors=len(communities))
    for community_id, community in enumerate(communities):
        size = len(community)
        for node in community:
            node_to_community[node] = community_id
        community_to_color[community_id] = community_colors[community_id]

    node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}


    for i, l in enumerate(label):
        for community_id, community in enumerate(communities):
            if l in community:
                community_label[i] = community_id

    community_labels.append(community_label)


    plt.figure()
    Graph(G,
          node_color=node_color, node_edge_width=0, node_labels=dict(zip(nodelist, nodelist)),
          edge_alpha=0.4, node_size=5, arrows=True, node_alpha=0.8,
          node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
          edge_layout='bundled', edge_layout_kwargs=dict(k=2000),edge_width=dict(zip(edgeList, LWidths))
          )

    plt.title("Community {}-{}".format(t, v))
    plt.box('off')
    plt.savefig(os.path.join(cfg['project_path'], "Com_Iso_" + str(n_cluster) + v + '.png'))
    plt.close()

#%% plotting motif usage and transition matrix

fig, axs = plt.subplots(2, 2, figsize=(10,10))
from scipy import stats
from scipy.stats import ks_2samp
ks_2samp(Motif_usages[0], Motif_usages[1])
p = stats.ttest_ind(Motif_usages[0], Motif_usages[1])
for j, videos in enumerate([control_videos, BD_videos]):
    # Plot histogram
    n = N[j]
    motif = Motif_usages[j]
    latent_vec = Latent_vectors[j]
    label = Labels[j]
    tm = TM[j]
    axs[0,j].bar(np.arange(0, np.shape(tm)[0] , 1), motif/len(label),
            width = 0.4)
    axs[0,j].set_title(titles[j])
    axs[0,j].set_xlabel("motifs")
    axs[0,j].set_ylabel("occurrence (%)")

    # Plot transition matrix
    im = axs[1,j].imshow(tm,vmin=0, vmax=0.42)
    axs[1,j].set_title("{} Transition matrix".format(titles[j]))
    axs[1,j].set_xticks(np.arange(0, np.shape(tm)[0] , 5))
    axs[1, j].set_yticks(np.arange(0,np.shape(tm)[0], 5))
    plt.colorbar(im, ax=axs[1, j])
    # Plot markov chain
plt.savefig(os.path.join(cfg['project_path'], "2Groups_" + str(n_cluster)  + '.png'))
plt.show()

#%% Plot visualization
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

targets = ['0','1','2','3','4','5,','6','7','8','9']
fig = plt.figure(figsize=(20,30))
colors = sns.color_palette()
embeds = [[],[]]
for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    motif = Motif_usages[j]
    latent_vec = Latent_vectors[j]
    label = Labels[j]
    # tsne_embed = TSNE(n_components=2, learning_rate='auto',
    #                    init='random').fit_transform(latent[:n, :])
    embed = umap.UMAP(n_components=3, min_dist=0.1, n_neighbors=200,
                         random_state=42).fit_transform(latent_vec)
    embeds[j] = embed
#%%
fig = plt.figure(figsize=(20,30))
for j, videos in enumerate([control_videos, BD_videos]):
    n = N[j]
    motif = Motif_usages[j]
    latent_vec = Latent_vectors[j]
    label = Labels[j]
    # tsne_embed = TSNE(n_components=2, learning_rate='auto',
    #                    init='random').fit_transform(latent[:n, :])
    embed = embeds[j]
    print("Embedding %d data points.." % n)

    ax = fig.add_subplot(3, 2, j + 1)
    for g in np.unique(label):
        i = np.where(label == g)
        ax.scatter(latent_vec[i, 0], latent_vec[i, 1], c=i, cmap='Spectral', s=2, alpha=1, label='%d' % g)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.scatter(cluster_center[:, 0], cluster_center[:, 1], c='black', s=40, alpha=0.5);

    ax.set_title("K-means  of {}".format(titles[j]))


    ax = fig.add_subplot(3, 2, j+3)
    num = np.unique(l)
    for g in np.unique(label):
        i = np.where(label == g)
        ax.scatter(embed[i, 0], embed[i, 1], c=i, cmap='Spectral', s=2, alpha=1, label='%d' % g)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title("UMAP of {}".format(titles[j]))

    # PCA
    pca = PCA(n_components=2).fit_transform(latent_vec[:n, :])
    principalDf = pd.DataFrame(data=pca)
    finalDf = pd.concat([principalDf, pd.DataFrame(label[:n])], axis=1)
    finalDf.columns = ['pc 1', 'pc 2','target']
    ax = fig.add_subplot(3, 2, j+5)
    for g in np.unique(label):
        i = np.where(label == g)
        ax.scatter(pca[i, 0], pca[i, 1], c=i, cmap='Spectral', s=2, alpha=1, label='%d' % g)
    # for target, color in zip(targets,colors):
    #     color = np.asarray(color)
    #     indicesToKeep = finalDf['target'] == target
    #     axs[1, j].scatter(finalDf.loc[indicesToKeep, 'pc 1'],
    #                       finalDf.loc[indicesToKeep, 'pc 2'],
    #                       color=color,s=2)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('PCA of {}'.format(titles[j]))
plt.show()

#%%
plt.figure()
x = np.arange(0,10,1)
y = np.ones(10)
plt.scatter(x,y, c=x,cmap='Spectral',label=x)
plt.legend()
plt.show()
#%%
from os import listdir
from os.path import isfile, join
mypath = r"G:\hBPM_Videos"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#%%

from itertools import zip_longest
d = onlyfiles
export_data = zip_longest(*d, fillvalue = '')
import csv
with open( r'G:\hBPM_Videos\video_name.csv', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(onlyfiles)


