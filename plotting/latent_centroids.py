# Created by zhanq at 5/3/2024
# File:
# Description:
# Scenario:
# Usage
#%% We also need the centroid of each subject, for error bars

# First, Get the centroids of each epoch of each population in latent space.
centroids = []
# [list of len 60 x 10] (10 motifs x 3 epoch x 2 population ) x (10 mean on zdim)

centroids_subjects = []
# [list of len 1500 x 10] (3 epoch x 2 pop x 25 sub/pop x 10 motifs) x 10 zdim
volume_subjects = []
# [list of 1500 x 10] (3 epoch x 2 pop x 25 sub/pop x 10 motifs) x 1
latent_all = []  # epoch 1 control, epoch 1 bd, epoch 2 control, epoch 2 bd, ...
label_all = []

centroids_reshape = np.zeros((3, 2, n_cluster, zdim))
for g in range(n_cluster):
    for epoch in range(1, 4):
        # for each population, get the centroid of each epoch
        for j, videos in enumerate([control_videos, BD_videos]):
            latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
            len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
            label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
            latent_all.append(latent_vec_epoch_pop)  # (6, 225000, 10)
            label_all.append(label_epoch_pop)  # (6, 225000)
            idx_g = np.where(label_epoch_pop == g)[0]
            latent_this_state_g = latent_vec_epoch_pop[idx_g, :]
            if len(latent_this_state_g):
                centroid = np.nanmean(latent_this_state_g, axis=0)
            else:
                centroid = np.full([zdim, ], np.nan)
            centroids.append(centroid)
            centroids_reshape[epoch - 1, j, g, :] = centroid

# Then, Get the centroids of each epoch of each subject in latent space.
for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        latent_vec_epoch_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        len_latent_each_epoch_each_population = len(latent_vec_epoch_pop)
        label_epoch_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for sub in range(n_subject_in_population):
            latent_this_sub = latent_vec_epoch_pop[9000*sub: 9000*(sub +1)] #9000 x 10
            label_this_sub = label_epoch_pop[9000*sub : 9000*(sub +1)]
            for g in range(n_cluster):
                idx_g = np.where(label_this_sub == g)[0]
                latent_this_state_g_this_sub = latent_this_sub[idx_g, :]
                if len(latent_this_state_g_this_sub):
                    latent_volume_this_state_g_this_sub = np.trace(np.cov(latent_this_state_g_this_sub.T))
                    centroid_sub = np.mean(latent_this_state_g_this_sub, axis=0)
                else:
                    print(f"a empty latent volume for {videos[sub]} in motif{g} epoch{epoch}")
                    latent_volume_this_state_g_this_sub = 0
                    centroid_sub = np.full([zdim,], 0)

                centroids_subjects.append(centroid_sub)
                volume_subjects.append(latent_volume_this_state_g_this_sub)



#%%  Below, we need the distance between centroids

# %% Compute centroids of motif-volume per person, and per population
person_centroids = np.empty(
    (3, 2, n_subject_in_population, n_cluster, zdim))  # epoch x pop x subject x 10 motifs x zdim
person_volumes = np.empty((3, 2, n_subject_in_population, n_cluster))
d_zit_BD_HP_ = np.empty((n_cluster, 3, n_subject_in_population * n_subject_in_population))
d_zit_HP_HP_ = np.empty((n_cluster, 3, n_subject_in_population * n_subject_in_population))
d_zit_BD_BD_ = np.empty((n_cluster, 3, n_subject_in_population * n_subject_in_population))


population_centroids = np.empty((3, 2, n_cluster, zdim))
population_volumes = np.empty((3, 2, n_cluster))
for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        latent_vec_this_epoch_this_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        label_this_epoch_this_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for g in range(n_cluster):
            d_zit_HP_HP = []
            d_zit_BD_BD = []
            for sub_i in range(n_subject_in_population):
                latent_vec_this_epoch_this_pop_this_person = latent_vec_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                label_this_epoch_this_pop_this_person = label_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                idx_g = np.where(label_this_epoch_this_pop_this_person == g)[0]
                latent_vec_this_epoch_this_pop_this_person_this_motif = latent_vec_this_epoch_this_pop_this_person[
                    idx_g]
                for sub_j in range(n_subject_in_population):
                    latent_vec_this_epoch_this_pop_another_person = latent_vec_this_epoch_this_pop[
                                                                 9000 * sub_j: 9000 * (sub_j + 1)]
                    label_this_epoch_this_pop_another_person = label_this_epoch_this_pop[9000 * sub_j: 9000 * (sub_j + 1)]
                    idx_g_2 = np.where(label_this_epoch_this_pop_another_person == g)[0]
                    latent_vec_this_epoch_this_pop_another_person_this_motif = latent_vec_this_epoch_this_pop_another_person[
                        idx_g_2]

                    inter_population_distance = dtw.distance(latent_vec_this_epoch_this_pop_this_person_this_motif, latent_vec_this_epoch_this_pop_another_person_this_motif)
                    if j == 0:
                        d_zit_HP_HP.append(inter_population_distance)
                    else:
                        d_zit_BD_BD.append(inter_population_distance)

                if len(latent_vec_this_epoch_this_pop_this_person_this_motif) > 1:
                    person_centroid = np.nanmean(latent_vec_this_epoch_this_pop_this_person_this_motif, axis=0)
                    person_volume = np.trace(np.cov(latent_vec_this_epoch_this_pop_this_person_this_motif.T))
                else:
                    person_centroid = np.full([zdim, ], np.nan)
                    person_volume = 0

                person_centroids[epoch - 1, j, sub_i, g, :] = person_centroid
                person_volumes[epoch - 1, j, sub_i, g] = person_volume
            if j == 0:
                d_zit_HP_HP_[g, epoch-1, :] = d_zit_HP_HP
            else:
                d_zit_BD_BD_[g, epoch - 1, :] = d_zit_BD_BD
for epoch in range(1, 4):
    for g in range(n_cluster):
        for sub_i in range(n_subject_in_population):
            for j, videos in enumerate([control_videos, BD_videos]):
                latent_vec_this_epoch_this_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
                label_this_epoch_this_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)

                latent_vec_this_epoch_this_pop_this_person = latent_vec_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                label_this_epoch_this_pop_this_person = label_this_epoch_this_pop[9000 * sub_i: 9000 * (sub_i + 1)]
                idx_g = np.where(label_this_epoch_this_pop_this_person == g)[0]
                latent_vec_this_epoch_this_pop_this_person_this_motif = latent_vec_this_epoch_this_pop_this_person[
                    idx_g]



for epoch in range(1, 4):
    for j, videos in enumerate([control_videos, BD_videos]):
        latent_vec_this_epoch_this_pop = np.concatenate(eval('Epoch{}_latent_vector'.format(epoch))[j], axis=0)
        label_this_epoch_this_pop = np.concatenate(eval('Epoch{}_labels'.format(epoch))[j], axis=0)
        for g in range(n_cluster):
            idx_g1 = np.where(label_this_epoch_this_pop == g)[0]
            latent_vec_this_epoch_this_pop_this_motif = latent_vec_this_epoch_this_pop[idx_g1]
            if len(latent_vec_this_epoch_this_pop_this_motif):
                population_centroid = np.nanmean(latent_vec_this_epoch_this_pop_this_motif, axis=0)
                population_volume = np.trace(np.cov(latent_vec_this_epoch_this_pop_this_motif.T))
            else:
                population_centroid = np.full([zdim, ], np.nan)
                population_volume = 0
            population_centroids[epoch - 1, j, g, :] = population_centroid
            population_volumes[epoch - 1, j, g] = population_volume

# Note: there is a small offset
print(np.nanmean(person_centroids[1, 0, :, 0, :], axis=0))
print(population_centroids[1, 0, 0, :])