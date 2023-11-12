#%% 
import csv
import os
import numpy as np
from collections import Counter


#%%
videos = ["BC1AASA", "BC1ADPI", "BC1ALKA", "BC1ALPA", "BC1ALRO", "BC1ANBU", "BC1ANGA", "BC1ANHE", 
            "BC1ANWI", "BC1ASKA", "BC1ATKU", "BC1BRBU", "BC1BRPO", "BC1BRSC", "BC1CERO", "BC1CISI", 
            "BC1COGR", "BC1DAAR", "BC1DEBR", "BC1DOBO", "BC1FEMO", "BC1GESA", "BC1GRLE", "BC1HAKO", 
            "BC1HETR", "BC1JACL", "BC1JECO", "BC1JUPA", "BC1JUST", "BC1KEMA", "BC1LABO", "BC1LACA", 
            "BC1LESA", "BC1LOKE", "BC1LOMI", "BC1LUOR", "BC1LUSE", "BC1MAMA", "BC1MEMA", "BC1MISE", 
            "BC1MOKI", "BC1NITA", "BC1OKBA", "BC1REFU", "CASH1", "GRJO1", "HESN1", "JEPT1", "JETH1", "MIRU1"]

path = r'C:\Users\kietc\SURF\jack-data\S3D\s3d_labels'

motifs = set() # should we take every motif into account?

def loadLabels(path):
    labels = {}
    for v in videos:
        labels[v] = [0]*401
        fname = "s3d_labels_{}.npy".format(v)
        counter = Counter(np.load(os.path.join(path, fname)).tolist())
        for i in counter:
            labels[v][i] = counter[i]
    return labels

def saveMotifUsage()