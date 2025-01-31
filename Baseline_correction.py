import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define the power law function
def power_law(I, I_c, n):
    V_c = 2.05  # 20.5 microvolts
    return V_c * (I / (I_c + 1e-9)) ** n  # Add a small offset to I_c to prevent division by zero

def normalize_dataset(data):
    """Fit a linear baseline between 0 and 20A and normalize the data."""
    mask = (data[:, 0] > 0) & (data[:, 0] <= 20)
    baseline_data = data[mask]

    # Fit a linear baseline
    if len(baseline_data) > 1:
        slope, intercept, _, _, _ = linregress(baseline_data[:, 0], baseline_data[:, 1])
    else:
        slope, intercept = 0, 0

    baseline = slope * data[:, 0] + intercept
    normalized_voltage = data[:, 1] - baseline
    normalized_data = np.column_stack((data[:, 0], normalized_voltage))

    return normalized_data

def adjust_text_position(texts, ax):
    """Adjust the position of text annotations to minimize overlap."""
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, only_move={'points': 'y', 'text': 'y'}, arrowprops=dict(arrowstyle='-', color='gray'))

def process_and_plot_with_fit(directory):
    data_by_angle = {}
    cmap = cm.plasma  # Use the plasma colormap for better contrast
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            with open(filepath, "r") as file:
                lines = file.readlines()

                angle = None
                field = None

                for line in lines:
                    if "Angle (deg.)" in line:
                        angle = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                    if "field / T" in line:
                        field = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                    if angle is not None and field is not None:
                        break

                start_data = False
                current_data = []

                for line in lines:
                    if start_data:
                        try:
                            current_data.append([float(x) for x in line.split()])
                        except ValueError:
                            continue
                    if "Current / A" in line:
                        start_data = True

                current_data = np.array(current_data)

                normalized_data = normalize_dataset(current_data)

                if angle not in data_by_angle:
                    data_by_angle[angle] = []
                data_by_angle[angle].append((field, normalized_data))

    for angle, datasets in data_by_angle.items():
        plt.figure(figsize=(10, 6))
        texts = []  # To store text objects for adjustment

        num_datasets = len(datasets)

        for i, (field, data) in enumerate(sorted(datasets, reverse=True)):
            positive_mask = data[:, 0] > 0
            I = data[positive_mask, 0]
            V = data[positive_mask, 1]

            try:
                popt, _ = curve_fit(power_law, I, V, p0=[10, 2])
                I_c, n = popt

                line_color = cmap(i / (num_datasets - 1))  # Uniformly index colormap between 0 and 1

                plt.plot(
                    data[:, 0],
                    data[:, 1],
                    "o",
                    markersize=1,
                    label=f"Field {field} T: n={n:.2f}, I_c={I_c:.2f} A",
                    color=line_color,
                )
                plt.plot(
                    data[:, 0],
                    power_law(data[:, 0], *popt),
                    "-",
                    linewidth=3,
                    color=line_color,
                )
                text = plt.text(
                    data[:, 0].max() * 0.95,
                    power_law(data[:, 0].max() * 0.95, *popt),
                    f"{field} T",
                    fontsize=15,
                    color="black",
                )
                texts.append(text)
            except RuntimeError:
                line_color = cmap(i / (num_datasets - 1))
                plt.plot(
                    data[:, 0],
                    data[:, 1],
                    "o",
                    label=f"Field {field} T (Fit Failed)",
                    color=line_color,
                )

        ax = plt.gca()
        adjust_text_position(texts, ax)

        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.title(f"Normalized Data with Fits at Angle: {angle}°", fontsize=16)
        plt.xlabel("Current (A)", fontsize=14)
        plt.ylabel("Normalized Voltage (uV)", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"angle_{angle}_fit_plot.png")
        plt.show()

data_directory = "data"
process_and_plot_with_fit(data_directory)
