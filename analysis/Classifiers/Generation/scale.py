#%% 
import csv
import numpy as np

#%% create dataframe
diagnosticPath = r"C:\Users\kietc\SURF\jack-data\diagnostic_data.csv"

dataframe = {}
with open(diagnosticPath) as file:
    csvreader = csv.reader(file, delimiter=',')
    next(csvreader)
    for row in csvreader:
        dataframe[row[0]] = [int(row[-1])]
        gender = 0
        if row[3] == 'F':
            gender = 1
        vec = np.array(row[1:3]).astype(float).tolist()
        vec[0] /= 60
        vec[1] /= 52
        vec.append(gender)
        dataframe[row[0]].extend(vec)

#%% gerenate csv
outPath = r"C:\Users\kietc\SURF\jack-data\scaled_diagnostic_data.csv"
header = ["video", "BD", "YMRS", "HAMD", "gender"]
with open(outPath, 'w') as csvfile:
    csvwriter = csv.writer(csvfile, lineterminator="\n")
    csvwriter.writerow(header)

    for v in dataframe:
        row = [v]
        row.extend(dataframe[v])
        csvwriter.writerow(row)

# %%
