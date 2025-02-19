# Created by zhanq at 2/18/2025
# File:
# Description:
# Scenario:
# Usage
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

from plotting.get_paths import get_my_path

# %%
myPath = get_my_path()
onedrive_path = myPath['onedrive_path']
github_path = myPath['github_path']
data_path = myPath['data_path']
# Load the data
file_path = rf'{onedrive_path}\Data\Behavior_VAE_data\demographic_table.csv'  # Replace with your file path
demographic_data = pd.read_csv(file_path)

demographic_data = demographic_data[~demographic_data['condition'].isin(['JUPA', 'LUSE'])]

# Preprocessing: Convert 'education level' to numeric
demographic_data['education level'] = demographic_data['education level'].replace('17 or more', 17).astype(float)
demographic_data['condition'] = demographic_data['condition'].replace(
    ['BD', 'BD1', 'BD2', 'Cyclothymic'], 'BD'
)

# Split data into HC and BD groups
hc_group = demographic_data[demographic_data['condition'] == 'HC']
bd_group = demographic_data[demographic_data['condition'].str.contains('BD')]

# Initialize results dictionary
results = {}

# Test 1: Age (Continuous using ANOVA)
age_stat, age_p = ttest_ind(hc_group['age'], bd_group['age'])
results['age'] = {'statistic': age_stat, 'p_value': age_p}

# Test 2: Education Level (Continuous using ANOVA)
edu_stat, edu_p = ttest_ind(hc_group['education level'], bd_group['education level'])
results['education level'] = {'statistic': edu_stat, 'p_value': edu_p}

# Test 3: Gender (Categorical)
gender_contingency = pd.crosstab(demographic_data['gender'], demographic_data['condition'])
gender_chi2, gender_p, _, _ = chi2_contingency(gender_contingency)
results['gender'] = {'chi2': gender_chi2, 'p_value': gender_p}

# Test 4: Race (Categorical)
race_contingency = pd.crosstab(demographic_data['race'], demographic_data['condition'])
race_chi2, race_p, _, _ = chi2_contingency(race_contingency)
results['race'] = {'chi2': race_chi2, 'p_value': race_p}

# Test 5: Ethnicity (Categorical)
ethnicity_contingency = pd.crosstab(demographic_data['ethnicity'], demographic_data['condition'])
ethnicity_chi2, ethnicity_p, _, _ = chi2_contingency(ethnicity_contingency)
results['ethnicity'] = {'chi2': ethnicity_chi2, 'p_value': ethnicity_p}

# Display the results as a DataFrame
results_df = pd.DataFrame(results).T
print(results_df)

# %%
# Calculate the mean age for BD and HC groups
mean_age_bd = bd_group['age'].mean()
mean_age_hc = hc_group['age'].mean()

# Print the results
print(f"Mean Age for BD group: {mean_age_bd}")
print(f"Mean Age for HC group: {mean_age_hc}")
# %%
import matplotlib.pyplot as plt

# Plot the age distribution for BD and HC groups
plt.figure(figsize=(10, 6))

# Plot histograms
plt.hist(hc_group['age'], bins=15, alpha=0.7, label='HC', color='blue', edgecolor='black')
plt.hist(bd_group['age'], bins=15, alpha=0.7, label='BD', color='orange', edgecolor='black')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution for BD and HC Groups')
plt.legend()

# Show the plot
plt.show()
