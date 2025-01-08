import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Define the power law function
def power_law(I, I_c, n):
    V_c = 2.05  # 20.5 microvolts
    return V_c * (I / (I_c + 1e-9)) ** n  # Add a small offset to I_c to prevent division by zero

def normalize_dataset(data):
    """Fit a linear baseline between 0 and 20A and normalize the data."""
    mask = (data[:, 0] > 0) & (data[:, 0] <= 20)  # Select data between 0 and 20A (positive currents only)
    baseline_data = data[mask]
    
    # Fit a linear baseline
    if len(baseline_data) > 1:  # Ensure sufficient points for fitting
        slope, intercept, _, _, _ = linregress(baseline_data[:, 0], baseline_data[:, 1])
    else:
        slope, intercept = 0, 0  # Fallback for insufficient data

    # Calculate baseline values for all currents
    baseline = slope * data[:, 0] + intercept
    
    # Normalize the data by subtracting the baseline
    normalized_voltage = data[:, 1] - baseline
    normalized_data = np.column_stack((data[:, 0], normalized_voltage))
    
    return normalized_data

def process_and_plot_with_fit(directory):
    # Create a dictionary to store data grouped by angle
    data_by_angle = {}

    # Iterate over all text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'r') as file:
                lines = file.readlines()

                # Extract angle and magnetic field from the header
                angle = None
                field = None

                for line in lines:
                    if "Angle (deg.)" in line:
                        angle = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                    if "field / T" in line:
                        field = float(re.search(r"[-+]?\d*\.\d+|\d+", line).group())
                    if angle is not None and field is not None:
                        break

                # Extract the numerical data
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

                # Normalize the dataset
                normalized_data = normalize_dataset(current_data)

                # Add the normalized data to the corresponding angle
                if angle not in data_by_angle:
                    data_by_angle[angle] = []
                data_by_angle[angle].append((field, normalized_data))

    # Plot normalized data for each angle with fitted curves
    for angle, datasets in data_by_angle.items():
        plt.figure(figsize=(10, 6))
        for field, data in datasets:
            # Filter positive currents for curve fitting
            positive_mask = data[:, 0] > 0
            I = data[positive_mask, 0]
            V = data[positive_mask, 1]

            # Perform curve fitting
            try:
                popt, _ = curve_fit(power_law, I, V, p0=[10, 2])
                I_c, n = popt
                fit_label = f"Fit: Field {field} T, n={n:.2f}, I_c={I_c:.2f} A"
                
                # Plot data points and fitted curve
                plt.plot(data[:, 0], data[:, 1], 'o',markersize=1, label=f"Data: Field {field} T")
                plt.plot(data[:, 0], power_law(data[:, 0], *popt), '-', linewidth=3, label=fit_label)
            except RuntimeError:
                plt.plot(data[:, 0], data[:, 1], 'o', label=f"Data: Field {field} T (Fit Failed)")

        plt.title(f"Normalized Data with Fits at Angle: {angle}°")
        plt.xlabel("Current (A)")
        plt.ylabel("Normalized Voltage (uV)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save or display each plot
        plt.savefig(f"angle_{angle}_fit_plot.png")  # Save plot as PNG
        plt.show()  # Display the plot

# Directory containing the data files
data_directory = "data"
process_and_plot_with_fit(data_directory)