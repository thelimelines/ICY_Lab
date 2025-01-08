import os
import re
import numpy as np
import matplotlib.pyplot as plt

def process_and_plot(directory):
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

                # Add the data to the corresponding angle
                if angle not in data_by_angle:
                    data_by_angle[angle] = []
                data_by_angle[angle].append((field, current_data))

    # Plot data for each angle
    for angle, datasets in data_by_angle.items():
        plt.figure(figsize=(10, 6))
        for field, data in datasets:
            plt.plot(data[:, 0], data[:, 1], label=f"Field: {field} T")
        plt.title(f"Angle: {angle}°")
        plt.xlabel("Current (A)")
        plt.ylabel("Voltage (uV)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save or display each plot
        plt.savefig(f"angle_{angle}_plot.png")  # Save plot as PNG
        plt.show()  # Display the plot

# Directory containing the data files
data_directory = "data"
process_and_plot(data_directory)
