#%% 
import csv
import glob

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
