# Created by zhanq at 5/23/2024
# File:
# Description:
# Scenario:
# Usage
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Sample list of labels
labels = [1, 3, 2, 2, 1, 3, 4, 1, 1, 1, 1, 1, 1, 1]
# Initialize the binary list with 0s
transitions = [0] * len(labels)

# Iterate through the list of labels and mark transitions
for i in range(1, len(labels)):
    if labels[i] != labels[i - 1]:
        transitions[i] = 1
# Define time steps
time_steps = np.arange(len(labels))

# Convert labels to a numpy array and reshape for KDE
labels_array = np.array(transitions).reshape(-1, 1)

# Perform kernel density estimation
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(labels_array)
log_density = kde.score_samples(labels_array)
intensity_estimates = np.exp(log_density)

# Normalize the intensity estimates to have a reasonable scale
intensity_estimates /= np.max(intensity_estimates)
intensity_estimates *= 5  # Scale the intensity to a reasonable range

# Generate Poisson-distributed events based on the estimated intensity function
poisson_events = np.random.poisson(intensity_estimates)
#%%
# Plot the intensity function and Poisson events
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Time step')
ax1.set_ylabel('Estimated Intensity', color=color)
ax1.step(time_steps, intensity_estimates, where='mid', color=color, label='Estimated Intensity')
ax1.tick_params(axis='y', labelcolor=color)

# Plot the Poisson events
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Poisson Events', color=color)
ax2.plot(time_steps, poisson_events, color=color, marker='o', linestyle='None', label='Poisson Events')

ax2.tick_params(axis='y', labelcolor=color)


# # Mark the change points
# for cp in change_points:
#     ax2.axvline(x=cp, color='r', linestyle='--', label='Change Point' if cp == change_points[0] else '')


fig.tight_layout()  # Ensure the plot labels do not overlap
plt.title('Estimated Intensity Function and Poisson Distributed Events')
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Given binary list of events
events = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]

# Convert to numpy array for convenience
event_array = np.array(events)

# Define time steps
time_steps = np.arange(len(events)).reshape(-1, 1)

# Find the indices where events occur
event_indices = np.where(event_array == 1)[0].reshape(-1, 1)

# Perform kernel density estimation on the event indices
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(event_indices)
log_density = kde.score_samples(time_steps)
intensity_estimates = np.exp(log_density)

# Normalize the intensity estimates if necessary
intensity_estimates /= np.max(intensity_estimates)
intensity_estimates *= np.mean(event_array) / np.mean(intensity_estimates)

# Plot the estimated intensity function
plt.figure(figsize=(10, 6))
plt.plot(time_steps, intensity_estimates, label='Estimated Intensity Function', color='blue')
plt.scatter(time_steps, event_array, label='Event Data', color='red', zorder=3)
plt.xlabel('Time Step')
plt.ylabel('Intensity / Event')
plt.title('Estimated Intensity Function and Event Data')
plt.legend()
plt.show()

