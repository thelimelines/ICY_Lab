import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from Baseline_correction import normalize_dataset, power_law, curve_fit

def extract_data(directory):
    data_points = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, "r") as file:
                lines = file.readlines()
                angle, field = None, None
                
                for line in lines:
                    if "Angle (deg.)" in line:
                        angle = float(line.split()[-1])
                    if "field / T" in line:
                        field = float(line.split()[-1])
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
                if len(current_data) == 0:
                    continue
                
                normalized_data = normalize_dataset(current_data)
                
                positive_mask = normalized_data[:, 0] > 0
                I = normalized_data[positive_mask, 0]
                V = normalized_data[positive_mask, 1]
                
                try:
                    popt, _ = curve_fit(power_law, I, V, p0=[10, 2])
                    I_c = popt[0]
                    data_points.append((angle, field, I_c))
                    data_points.append((180 - angle, field, I_c))  # Mirroring about 90 degrees
                except RuntimeError:
                    continue
    
    return np.array(data_points)

def plot_3d_mesh(data_points):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    angles = data_points[:, 0]
    fields = data_points[:, 1]
    critical_currents = data_points[:, 2]
    
    ax.plot_trisurf(fields, angles, critical_currents, cmap=cm.plasma, edgecolor='none')
    
    ax.set_xlabel("Magnetic Field (T)", fontsize=20, labelpad=15)
    ax.set_ylabel("Angle (°)", fontsize=20, labelpad=15)
    ax.set_zlabel("Critical Current (A)", fontsize=20, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='z', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(f"3D_plot.png")
    plt.show()

data_directory = "data"
data_points = extract_data(data_directory)
if len(data_points) > 0:
    plot_3d_mesh(data_points)
