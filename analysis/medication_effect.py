import re
from collections import Counter

import pandas as pd

from plotting.get_paths import get_my_path


def drop_rows_if_50(df):
    # Remove unqualified person rows from the csv,
    # which are the 27th and 36th rows JUPA and LUSE
    # if the length of the dataframe is 50
    if len(df) == 50:
        df.drop(index=[27, 36], inplace=True)
    return df


# Function to normalize medication names (remove dosage info)
def normalize_med_name(med_name):
    return re.sub(r"[\d]+ ?(mg|mcg|g|ml|units|iu)?", "", med_name, flags=re.IGNORECASE).strip()


# %%
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']

# Load the Excel file
file_path = f"{onedrive_path}\Behavior_VAE_data\Victoria MS subjects_with_meds.xlsx"
xls = pd.ExcelFile(file_path)

# Read the relevant sheet
df_video_info_bd = pd.read_excel(xls, sheet_name='video-information-subjects')
drop_rows_if_50(df_video_info_bd)
# Extract medications from BD group
bd_medications = df_video_info_bd["CurrentMedications"].dropna().str.lower().str.split("; ")

# Remove BC1LUSE from the BD group
df_video_info_bd_filtered = df_video_info_bd[df_video_info_bd["video_name"] != "BC1LUSE"]

# Extract medications again from the filtered BD group
bd_medications_filtered = df_video_info_bd_filtered["CurrentMedications"].dropna().str.lower().str.split("; ")

# Further split multi-medication entries into individual medications
all_meds_individual = []
for sublist in bd_medications_filtered:
    for med in sublist:
        all_meds_individual.extend(med.split(", "))  # Split combinations

# Recount occurrences of individual medications
med_count_final = Counter([med.strip() for med in all_meds_individual])

# Remove "none" if present
med_count_final.pop("none", None)

# Get the top 5 most common individual medications
top_5_final_meds = [med for med, _ in med_count_final.most_common(5)]

# Normalize all medication names
normalized_meds = [normalize_med_name(med) for med in all_meds_individual]

# Recount occurrences of normalized medications
med_count_normalized = Counter(normalized_meds)

# Remove "none" if present
med_count_normalized.pop("none", None)

# Get the top 5 most common individual medications (normalized)
top_5_normalized_meds = [med for med, _ in med_count_normalized.most_common(5)]

top_5_normalized_meds
# %%
# Create a new dataframe with subject names
df_med_usage = pd.DataFrame()
df_med_usage["subject_name"] = df_video_info_bd_filtered["video_name"]
df_med_usage["is_BD"] = df_video_info_bd_filtered["condition"]


# Function to normalize medication names (remove dosage and time descriptors)
def normalize_med_name(med_name):
    # Remove dosage (e.g., 300 mg, 5 mg, 100 ml)
    med_name = re.sub(r"[\d]+ ?(mg|mcg|g|ml|units|iu|tablets|capsules)?", "", med_name, flags=re.IGNORECASE)
    # Remove time-related descriptors (e.g., "in morning", "evening", "PM", "night")
    med_name = re.sub(r"(in morning|evening|at night|pm|am|-pm|-am|daily|morning|night)", "", med_name,
                      flags=re.IGNORECASE)
    return med_name.strip()


# Function to check if any top medication appears in the subject's medication list
def takes_medication(med_list, med_name):
    if pd.isna(med_list):
        return 0
    # Normalize and split medications using multiple delimiters (; and ,)
    normalized_meds = [normalize_med_name(med.strip()) for med in re.split(r"; |, ", med_list.lower())]
    # Check if any normalized top medication appears in the subject's medication list
    return any(med_name in med for med in normalized_meds)


# Normalize the top 5 medication names as well
top_5_normalized_meds = [normalize_med_name(med) for med in top_5_normalized_meds]

# Add columns for the top 5 most common medications (fully normalized)
for med in top_5_normalized_meds:
    df_med_usage[med] = df_video_info_bd_filtered["CurrentMedications"].apply(lambda x: takes_medication(x, med))
# Ensure all values in the DataFrame are strictly 0 or 1
# Ensure only the medication columns (lamictal, abilify, seroquel, lithium, zoloft) are converted to 0/1
med_columns = top_5_normalized_meds  # These are the medication columns
df_med_usage[med_columns] = df_med_usage[med_columns].astype(int)
data_path = f"{onedrive_path}\Behavior_VAE_data\medication_list.csv"
df_med_usage.to_csv(data_path, index=False)
