def compute_l0_entropy(transition_m, last_state):
    # https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
    invertible_T = add_self_transition(transition_m, last_state)
    if len(invertible_T):
        S, U = scipy.linalg.eig(invertible_T.T)
        stationary = np.array(U[:, np.where(np.abs(S - 1.) < 1e-8)[0][0]].flat)
        stationary = stationary / np.sum(stationary)
        stationary = stationary.real
        H = scipy.stats.entropy(invertible_T, base=2, axis=1)
        entropy = stationary.dot(H)
    else:
        entropy = 0
    return entropy


#%% 
import csv
import os
import glob
import numpy as np

#%% config
path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\\10_km_label_{}.npy'
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

#%%
for j, videos in enumerate([control_videos, BD_videos]):
    n = 0
    for i in range(len(videos)):
        v = videos[i]
        print("Loading {} data...".format(v))
        label = np.load(
            r'Behavior_VAE_data\BD20-Jun5-2022\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(v, n_cluster, n_cluster, v))
        transition_m = np.load(
            r'{}\Behavior_VAE_data\BD20-Jun5-2022\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, v))
        cluster_center = np.load(
            r'{}\Behavior_VAE_data\BD20-Jun5-2022\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path,
                                                                                              project_name, v,
                                                                                              n_cluster, v))
        motif_usage = np.load(
            r'{}\Behavior_VAE_data\BD20-Jun5-2022\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name,
                                                                                           v, n_cluster, v))
    print(cluster_center)

#%% Generate dv_data.csv
out_file = "../../../SURF/dv_data.csv"

# writing to csv file
with open(out_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, lineterminator='\n')
    csvwriter.writerow(['', 'm0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9'])
    for j, videos in enumerate([control_videos, BD_videos]):
        n = 0
        for i in range(len(videos)):
            v = videos[i]
            print("Loading {} data...".format(v))
            label = np.load(
                r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path, project_name,
                                                                                               v, n_cluster, n_cluster, v))
            transition_m = np.load(
                r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(
                    onedrive_path, project_name, v, n_cluster, v))
            cluster_center = np.load(
                r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path,
                                                                                                  project_name, v,
                                                                                                  n_cluster, v))
            motif_usage = np.load(
                r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path, project_name,
                                                                                               v, n_cluster, v))
            vec = motif_usage.tolist()
            vec.insert(0, v)
            csvwriter.writerow(vec + transition_m.flatten().tolist())

#%%
import csv
#%% Generate dv_datat_entropy.csv
out_file = "../../../SURF/dv_data_entropy.csv"

# writing to csv file
with open(out_file, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, lineterminator='\n')
    csvwriter.writerow(['', 'entropy', 'num_zero_row', 'num_one_item', 'num_zero_item'])
    for v in videos:
        YMRS_score.append(YMRS[v][0])
        HAM_D_score.append(HAM_D[v][0])
        print("Loading {} data...".format(v))

        label = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\{}_km_label_{}.npy'.format(onedrive_path,
                                                                                            project_name, v,
                                                                                            n_cluster, n_cluster, v))
        transition_m = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\community\transition_matrix_{}.npy'.format(
                onedrive_path, project_name, v, n_cluster, v))
        cluster_center = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\cluster_center_{}.npy'.format(onedrive_path,
                                                                                                project_name, v,
                                                                                                n_cluster, v))
        motif_usage = np.load(
            r'{}\Behavior_VAE_data\{}\results\{}\VAME\kmeans-{}\motif_usage_{}.npy'.format(onedrive_path,
                                                                                            project_name, v,
                                                                                            n_cluster, v))
        folder = os.path.join(cfg['project_path'], "results", v, model_name, 'kmeans-' + str(n_cluster), "")
        latent_vector = np.load(os.path.join(folder, 'latent_vector_' + v + '.npy'))  # L x 30

        transition = transition_m.copy()

        transition_matrices.append(transition_m)

        # Entropy, and classic metrics of transition matrix
        num_zero_row, num_one_item, num_zero_item = count_zeros(transition_m)
        entropy = compute_l0_entropy(transition_m, label[-1])
        vec = [entropy, num_zero_row, num_one_item, num_zero_item]
        vec.insert(0, v)
        csvwriter.writerow(vec)
