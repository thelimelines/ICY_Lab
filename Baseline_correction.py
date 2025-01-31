import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def power_law(I, I_c, n):
    V_c = 2.05  # 20.5 microvolts
    return V_c * (I / (I_c + 1e-9)) ** n  # Add small offset to I_c to prevent division by zero

def normalize_dataset(data):
    """Fit a linear baseline between 0 and 20A and normalize the data."""
    mask = (data[:, 0] > 0) & (data[:, 0] <= 20)
    baseline_data = data[mask]

    if len(baseline_data) > 1:
        slope, intercept, _, _, _ = linregress(baseline_data[:, 0], baseline_data[:, 1])
    else:
        slope, intercept = 0, 0

    baseline = slope * data[:, 0] + intercept
    normalized_voltage = data[:, 1] - baseline
    return np.column_stack((data[:, 0], normalized_voltage))

def adjust_text_position(texts, ax):
    from adjustText import adjust_text
    adjust_text(texts, ax=ax, only_move={'points': 'y', 'text': 'y', 'objects': 'x'}, expand_points=(1.2, 1.5),
                force_text=(0.2, 0.5), arrowprops=dict(arrowstyle='-', color='gray'))

def process_and_plot_with_fit(directory, show_labels=True, font_size=20):
    data_by_angle = {}
    cmap = cm.plasma  
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                lines = file.readlines()
                angle, field = None, None
                
                for line in lines:
                    if "Angle (deg.)" in line:
                        angle = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                    if "field / T" in line:
                        field = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                    if angle is not None and field is not None:
                        break
                
                start_data, current_data = False, []
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
        texts = []
        ic_texts = []
        num_datasets = len(datasets)

        for i, (field, data) in enumerate(sorted(datasets, reverse=True)):
            positive_mask = data[:, 0] > 0
            I, V = data[positive_mask, 0], data[positive_mask, 1]
            try:
                popt, _ = curve_fit(power_law, I, V, p0=[10, 2])
                I_c, n = popt
                line_color = cmap(i / (num_datasets - 1))
                
                plt.plot(data[:, 0], data[:, 1], "o", markersize=1, color=line_color)
                plt.plot(data[:, 0], power_law(data[:, 0], *popt), "-", linewidth=3, color=line_color, label=rf"{field} T, $I_c={I_c:.2f}$ A")
                
                plt.axhline(y=power_law(I_c, *popt), linestyle="--", color="gray", linewidth=2)
                
                if show_labels:
                    text = plt.text(
                        data[:, 0].max() * 1.05,
                        power_law(data[:, 0].max() * 1.05, *popt) + 2.0,
                        f"{field} T",
                        fontsize=font_size,
                        color="black",
                    )
                    texts.append(text)
                    
                    ic_text = plt.text(
                        I_c + 5, power_law(I_c, *popt) + 2.0,
                        f"{I_c:.0f} A",
                        fontsize=font_size,
                        color="black"
                    )
                    ic_texts.append(ic_text)
                
            except RuntimeError:
                line_color = cmap(i / (num_datasets - 1))
                plt.plot(data[:, 0], data[:, 1], "o", color=line_color)
        
        ax = plt.gca()
        if show_labels:
            adjust_text_position(texts, ax)
            adjust_text_position(ic_texts, ax)
        else:
            plt.legend(fontsize=font_size)
        
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        plt.title(f"Normalized Data with Fits at Angle: {angle}°", fontsize=font_size)
        plt.xlabel("Current (A)", fontsize=font_size)
        plt.ylabel("Normalized Voltage (μV)", fontsize=font_size)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"angle_{angle}_fit_plot.png")
        plt.show()

data_directory = "data"
process_and_plot_with_fit(data_directory, show_labels=False, font_size=20)
