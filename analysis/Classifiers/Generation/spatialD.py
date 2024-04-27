#%%

import numpy as np
from scipy.optimize import curve_fit


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
    center_necks[v] = [center_neck[:][15], center_neck[:][16]]


#%%
def save_vector_to_file(vector, filename):
    with open(filename, 'w') as file:
        file.write("Format: CenterX(mm) CenterY(mm)\n\n0\t0\n")

        for subject, (x_list, y_list) in vector.items():
            # Check if both x_list and y_list have 27000 values
            if len(x_list) == len(y_list) == 27000:
                # Iterate through corresponding x and y values
                for x, y in zip(x_list, y_list):
                    file.write(f"{x}\t{y}\n")  # Write the coordinates to the file
            else:
                print(f"Issue with {subject}: x_list and y_list must both have 27000 values.")


#%%
def spatialD(x, y, n, K):
    """
    Calculate spatial d parameters for a series of locations (x, y).

    Inputs:
    x, y:   Arrays of corresponding length representing (x, y) coordinates.
            Data points can be recorded at regular time intervals.
    n:      Number of data points to consider for calculating an incremental d.
    K:      Array of resolutions (e.g., K=[1,2,3,4]).

    Outputs:
    D:      Spatial d parameter averaged over the time interval of the recording.
    Dloc:   Array of all local D values calculated on intervals of n data points.
    Kloc:   Array of resolutions corresponding to local D values (K values used).
    Lloc:   Array of path length values corresponding to local D calculations.

    References:
    Paulus & Geyer (1991) and Higushi (1988)
    """

    Nk = len(K)
    Nx = len(x)
    NBinterv = Nx - n + 1

    if Nx < n:
        # If data chunk is not long enough
        D = 0  # Impossible value for d, continue running program
        Dloc = np.zeros(NBinterv)
        Kloc = np.zeros(NBinterv * Nk)
        Lloc = np.zeros(NBinterv * Nk)
        
    else:
        Dloc = np.zeros(NBinterv)
        Kloc = np.zeros(NBinterv * Nk)
        Lloc = np.zeros(NBinterv * Nk)

        for interv in range(NBinterv): # Loop on number of intervals studied over the whole initial xy-vector
            xx = x[interv:interv+n]
            yy = y[interv:interv+n]

            L = np.zeros(Nk)

            for it in range(Nk): #Loop on resolution  
                LLL = []
                k = K[it]

                # Calculate L(k) - path length with resolution k
                # (calculated using the average of the vectors of xy-data every k points on the interval - there are k such vectors)
                for i in range(1, k+1):
                    III = np.arange(i-1, n, k)
                    xxx = xx[III]
                    yyy = yy[III]
                    LLL.append(np.sum(np.sqrt(np.diff(xxx)**2 + np.diff(yyy)**2)) * ((n-1) / k) / np.floor((n-i) / k) / k)

                L[it] = np.sum(LLL) / k

                # Record local values
                index = interv * Nk + it
                Lloc[index] = L[it]
                Kloc[index] = k

            # Linear least squares fit
            P = np.polyfit(np.log(K), np.log(L), 1)
            slope = -P[0]
            if 0 <= slope < 3:
                Dloc[interv] = slope
            else:
                Dloc[interv] = 0

        # Average of terms greater than 0
        Dloc_positive = Dloc[Dloc > 0]
        if len(Dloc_positive) > 0:
            D = np.mean(Dloc_positive)
        else:
            D = 0

    return {
        'D': D,
        'Dloc': Dloc,
        'Kloc': Kloc,
        'Lloc': Lloc
    }


#%%
n = 27000

results = {}


for v in videos:
    results[v] = spatialD(center_necks[v][0], center_necks[v][1], )

#%% Main
    
what = 1  # what=0 => calculate parameters with and without artifacts
              # what=1 => calculate parameters only on data without artifacts
filter_cutoff = 5  # Hz
artifact_cutoff = 30  # pixels per 0.033s
nmin = 24  # minimum number of data points to create a valid chunk of data
denominator = 27000
f = 30  # Hz - frequency of data recording

K = [2**i for i in range(6)]  # Vector of resolutions for spatial-d calculation
n_spatialD = 2**5*2 + 1  # Multiple of all values in K - Number of data points to consider for an incremental Spatial-D calculation
beams = [96, 64]  # Number of beams in the x and y direction [beams_x, beams_y]
RoomDim = [720, 480]  # Dimension of the room in number of pixels [MaxPix_x, MaxPix_y]
RoomDim_width = 4.3  # in meters

DistTravelled_cutoff = 0.06  # Radius in meters defining purposeful movement (6cm)
DistTravelled_cutoff_pixels = int(DistTravelled_cutoff / RoomDim_width * RoomDim[0])  # in pixels

# Call activity function with specified parameters
Results = activity(what, xy_original, filter_cutoff, artifact_cutoff, nmin, denominator, f, n_spatialD, K, beams, RoomDim, DistTravelled_cutoff_pixels)

t = Results.t
xy = Results.xy

d = Results.d
d_filt = Results.d_filt
d_filt_clean = Results.d_filt_clean
td_clean = Results.td_clean

# xy after filter and artifact cleaning + beam conversion if relevant
x_filt_clean = Results.x_filt_clean  
y_filt_clean = Results.y_filt_clean
txy_clean = Results.txy_clean

# xy after filter and artifact cleaning + no beam conversion
x_filt_clean_nobeam = Results.x_filt_clean_nobeam  
y_filt_clean_nobeam = Results.y_filt_clean_nobeam

# xy after travelled-filter - no beam conversion
t_filt_clean_DT = Results.t_filt_clean_DT  # filtered using travelled cutoff
x_filt_clean_DT = Results.x_filt_clean_DT  # filtered using travelled cutoff
y_filt_clean_DT = Results.y_filt_clean_DT  # filtered using travelled cutoff

counts = Results.counts
counts_filt = Results.counts_filt
counts_filt_clean = Results.counts_filt_clean

# Spatial-D results
if what == 0:
    Dspa = Results.Dspa  # Average spatial-D for this subject - no data processing
    Dloc = Results.Dloc  # Vector of all spatial-D values calculated on interval increments of 12 data points

    Dspa_filt = Results.Dspa_filt  # Average spatial-D for this subject - after low pass filter
    Dloc_filt = Results.Dloc_filt

Dspa_filt_clean = Results.Dspa_filt_clean  # Average spatial-D for this subject - after low pass filter & artifact cleaning
Dspa_chunk = Results.Dspa_chunk
N_chunk = Results.N_chunk
Dloc_chunk = Results.Dloc_chunk
K_chunk = Results.Kchunk
L_chunk = Results.Lchunk

MovedDist = Results.MovedDist  # Distance Moved in terms of number of room x-length - filtered/non-artifacted signal
TravelledDist = Results.TravelledDist  # Distance Travelled in terms of number of room x-length - filtered/non-artifacted signal

