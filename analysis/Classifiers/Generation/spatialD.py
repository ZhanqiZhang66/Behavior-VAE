import numpy as np
from scipy.optimize import curve_fit




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
