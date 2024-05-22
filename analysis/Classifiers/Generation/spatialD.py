#%%

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.collections as mcollections



#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
                  "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
                  "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
                  "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
                  "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
                  "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

dlc_path = r'C:\Users\kietc\OneDrive - UC San Diego\Behavior_VAE_data\BD25-HC25-final-May17-2023\results\{}\VAME\kmeans-10\dlc_vector_{}.npy'
center_necks = {}

#%%
for v in videos:
    center_neck = np.load(dlc_path.format(v, v))
    center_necks[v] = [center_neck[:, 15], center_neck[:, 16]]


"""
START OF PLOTTING
"""

#%%
def plot_colored_by_time(ax, x_coords, y_coords, cmap_name):
    # Create the segments for the line
    points = np.array([x_coords, y_coords]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a LineCollection with a colormap
    norm = plt.Normalize(0, len(x_coords) - 1)
    lc = mcollections.LineCollection(segments, cmap=cmap_name, norm=norm)
    lc.set_array(np.arange(len(x_coords)))
    lc.set_linewidth(2)  # Adjust the linewidth for the line

    # Add the LineCollection to the plot
    ax.add_collection(lc)

#%%
motif1 = {
    'BC1AASA': [24267, 24363], # 3900
    'BC1ADPI': [19325, 19426], # 1032
    'BC1ALKA': [8550, 8639], # 804
    'BC1ALPA': [3742, 3855], # 942
    'BC1ALRO': [9255, 9344], # 822
}

#%%
clips = motif1
coords = []

for v in clips:
    coords.append([center_necks[v][0][clips[v][0]: clips[v][1]],center_necks[v][1][clips[v][0]: clips[v][1]]] )

#%%
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# List of colormaps for variety
colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'hsv', 'spring', 'summer', 'autumn']

# Plot each set of coordinates with a different colormap
for i, coordinates in enumerate(coords):
    x_coords = np.array(coordinates[0])
    y_coords = np.array(coordinates[1])
    plot_colored_by_time(ax, x_coords, y_coords, colormaps[i % len(colormaps)]) #

# Set the limits of the plot to match the room dimensions
ax.set_xlim(0, 720)
ax.set_ylim(0, 480)


# Add colorbars for each line collection
for i in range(1): #len(coords)
    cbar = plt.colorbar(mappable=ax.collections[i], ax=ax, orientation='vertical')
    cbar.set_label(f'Time for Coordinates {i+1}')

# Show the plot
plt.grid(True)
plt.show()

"""
END OF PLOTTING
"""



#%%
def save_vector_to_file(filename, vector):
    with open(filename, 'w') as file:
        file.write("Format: CenterX(mm) CenterY(mm)\n\n0\t0\n")
        for i in range(27000):
            file.write(f"{vector[0][i]}\t{vector[1][i]}\n")  # Write the coordinates to the file

#%%        
path = r'C:\Users\kietc\OneDrive - UC San Diego\SURF\TRC\{}_TRC.txt'
#%%
save_vector_to_file(path, center_necks[v])

#%%
for v in videos:
    save_vector_to_file(path.format(v), center_necks[v])




#%%
import scipy.io
import numpy as np
import csv


path = r"C:\Users\kietc\OneDrive - UC San Diego\Documents\MATLAB\Results\D"

# %%
def save_D(path, D):
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator="\n")
        header = ['video', 'spatialD']
        csvwriter.writerow(header)

        for v in videos:
            csvwriter.writerow([v, D[v]])

#%%
D = {}

for v in videos:
    mat_contents = scipy.io.loadmat(path + '\\' + v + '_D.mat')
    D[v] = mat_contents['tempD'][0][0]
# %%
path = r"C:\Users\kietc\OneDrive - UC San Diego\SURF\spatialD.csv"
save_D(path, D)
# %%
