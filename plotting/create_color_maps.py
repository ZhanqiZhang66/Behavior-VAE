# Created by zhanq at 4/26/2024
# File:
# Description:
# Scenario:
# Usage
import matplotlib.pyplot as plt
import colorsys
from matplotlib.colors import ListedColormap


# Generate 30 distinct colors
def generate_distinct_colors(num_colors):
    # Initialize an empty list to store colors
    colors = []

    # Generate colors in HSL space
    for i in range(num_colors):
        hue = i / num_colors  # Vary hue across the range
        saturation = 0.7  # Fixed saturation
        lightness = 0.6  # Fixed lightness
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)

    return ListedColormap(colors)








