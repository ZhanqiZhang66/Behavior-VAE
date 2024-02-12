
#%% 
import csv
import os
import numpy as np
from vame.analysis.community_analysis import read_config, compute_transition_matrices
import matplotlib.pyplot as plt
from utils import load_motif_labels, compute_l0_entropy, effective_num_states, load_tmatrices, save_tmatrices


#%%
def generate_matrices(labels, motifSize, split = 1):
    matrices = {}
    # frames = len(labels[next(iter(labels))][0])
    for v in videos:
        matrices[v] = []
        for i in range(split):
            matrix = compute_transition_matrices([v], [labels[v][i]], motifSize)
            matrices[v].append(matrix[0].tolist())
    return matrices

def generate_graphs(path, matrices, motifSize, split = 1):
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


def generate_count(matrices, split):
    count = {}
    for v in videos:
        count[v] = []
        for i in range(split):
            out = count_zeros(np.array(matrices[v][i]))[1]
            count[v].append(out)
    return count

def generate_ens(matrices, split):
    ens = {}
    for v in videos:
        ens[v] = []
        for i in range(split):
            ens[v].append(effective_num_states(matrices[v][i])[1])
    return ens

def generate_entropy(matrices, labels, split):
    entropy = {}
    for v in videos:
        entropy[v] = []
        for i in range(split):
            entropy[v].append(compute_l0_entropy(np.array(matrices[v][i]), labels[v][i][-1]))
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
vLabelPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
vMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix_3\{}_epoch_{}-{}.npy"

dLabelPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\DLC_10_km_label_{}.npy'
dMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\transition_matrix_3\{}_epoch_{}-{}.npy"

hLabelPath = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\score_labels_{}.npy'
hMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\transition_matrix_3\{}_epoch_{}-{}.npy"

sLabelPath = r'C:\Users\kietc\SURF\jack-data\S3D\s3d_labels\s3d_labels_{}.npy'
sMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\transition_matrix_3\{}_epoch_{}-{}.npy"

mLabelPath = r'C:\Users\kietc\SURF\jack-data\MMAction\mmaction_labels\mmaction_labels_{}.npy'
mMatrixPath = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\transition_matrix_3\{}_epoch_{}-{}.npy"



#%%
vLabels = load_motif_labels(vLabelPath, videos, 27000, 3)
dLabels = load_motif_labels(dLabelPath, videos, 27000, 3)
hLabels = load_motif_labels(hLabelPath, videos, 27000, 3)
sLabels = load_motif_labels(sLabelPath, videos, 27000, 3)
mLabels = load_motif_labels(mLabelPath, videos, 27000, 3)


#%%
#vMatrices = generate_matrices(vLabels, 10, split = 3)
dMatrices = generate_matrices(dLabels, 10, split = 3)
#hMatrices = generate_matrices(hLabels, 11, split = 3)
#sMatrices = generate_matrices(sLabels, 400, split = 3)
#mMatrices = generate_matrices(mLabels, 81, split = 3)

#%%
#save_tmatrices(vMatrixPath, videos, vMatrices, 3)
save_tmatrices(dMatrixPath, videos, dMatrices, 3)
#save_tmatrices(hMatrixPath, videos, hMatrices, 3)
#save_tmatrices(sMatrixPath, videos, sMatrices, 3)
#save_tmatrices(mMatrixPath, videos, mMatrices, 3)


#%%
vMatrices = load_tmatrices(vMatrixPath, videos, split = 3)
hMatrices = load_tmatrices(hMatrixPath, videos, split = 3)
sMatrices = load_tmatrices(sMatrixPath, videos, split = 3)
mMatrices = load_tmatrices(mMatrixPath, videos, split = 3)

#%%
vEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\entropy_3_split.csv'
dEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\entropy_3_split.csv'
hEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\entropy_3_split.csv'
sEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\entropy_3_split.csv'
mEntropyPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\entropy_3_split.csv'

#%%
#vEntropy = generate_entropy(vMatrices, vLabels, 3)
dEntropy = generate_entropy(dMatrices, dLabels, 3)
#hEntropy = generate_entropy(hMatrices, hLabels, 3)
#sEntropy = generate_entropy(sMatrices, sLabels, 3)
#mEntropy = generate_entropy(mMatrices, mLabels, 3)

#%%
#export_data(vEntropyPath, vEntropy, 3)
export_data(dEntropyPath, dEntropy, 3)
#export_data(hEntropyPath, hEntropy, 3)
#export_data(sEntropyPath, sEntropy, 3)
#export_data(mEntropyPath, mEntropy, 3)

#%%
dENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\ens_3_split.csv'
# %%
vENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\ens_3_split.csv'
dENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\ens_3_split.csv'
hENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\ens_3_split.csv'
sENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\ens_3_split.csv'
mENSPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\ens_3_split.csv'

#%%
dENS = generate_ens(dMatrices, 3)
#%%
vENS = generate_ens(vMatrices, 3)
hENS = generate_ens(hMatrices, 3)
sENS = generate_ens(sMatrices, 3)
mENS = generate_ens(mMatrices, 3)

# %%
export_data(dENSPath, dENS, 3)
# %%
export_data(vENSPath, vENS, 3)
export_data(hENSPath, hENS, 3)
export_data(sENSPath, sENS, 3)
export_data(mENSPath, mENS, 3)

# %%
dCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\DLC\count_3_split.csv'
# %%
vCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\VAME\count_3_split.csv'
hCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\hBPM\count_3_split.csv'
sCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\S3D\count_3_split.csv'
mCountPath = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\MMAction\count_3_split.csv'

#%%
dCount = generate_count(dMatrices, 3)
#%%
vCount = generate_count(vMatrices, 3)
hCount = generate_count(hMatrices, 3)
sCount = generate_count(sMatrices, 3)
mCount = generate_count(mMatrices, 3)

# %%
export_data(vCountPath, vCount, 3)
export_data(hCountPath, hCount, 3)
export_data(sCountPath, sCount, 3)
export_data(mCountPath, mCount, 3)



# %%
tm = np.array([[0.        , 0.28571429, 0.        , 0.14285714, 0.        ,
        0.        , 0.        , 0.42857143, 0.        , 0.14285714],
       [0.08333333, 0.        , 0.16666667, 0.08333333, 0.        ,
        0.        , 0.16666667, 0.        , 0.08333333, 0.41666667],
       [0.2       , 0.6       , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.2       , 0.        , 0.        ],
       [0.        , 0.4       , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.6       ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.25      , 0.75      , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.5       , 0.25      , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.25      ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.1       , 0.2       , 0.3       , 0.3       , 0.        ,
        0.        , 0.1       , 0.        , 0.        , 0.        ]])
compute_l0_entropy(tm[0], 9)
# %%
