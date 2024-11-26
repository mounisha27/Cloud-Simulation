                                                                    #Cloud Simulation with SimPy
import subprocess
import sys

# Function to check and install missing libraries
def install_libraries():
    # List of required libraries
    required_libraries = ['pandas', 'numpy', 'matplotlib', 'random']
    
    # Loop through each library and check if it is installed
    for library in required_libraries:
        try:
            # Try importing the library
            __import__(library)
        except ImportError:
            # If the library is not installed, install it using pip
            print(f"{library} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Installing the libraries before proceeding with the code execution
install_libraries()

# importing the libraries after installation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the dataset
df = pd.read_csv("cleaned_dataset.csv") 
df = df[['cpu_usage_distribution', 'memory_accesses_per_instruction', 'estimated_power_consumption', 'start_time', 'end_time']]

# Converting start_time and end_time to datetime format 
df['start_time'] = pd.to_datetime(df['start_time'])
df['end_time'] = pd.to_datetime(df['end_time'])

# Calculate the duration 
df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()

# Function to calculate energy consumption
def calculate_energy(cpu_needed, memory_needed, execution_time):
    # CPU and memory weightage factors to calculate energy consumption
    CPU_WEIGHT = 0.05  # Weight for CPU usage
    MEMORY_WEIGHT = 0.02  # Weight for memory usage
    # Energy formula based on CPU and memory usage, and execution time
    return cpu_needed * execution_time * CPU_WEIGHT + memory_needed * execution_time * MEMORY_WEIGHT

# Function to determine usage category dynamically based on CPU and memory usage
def get_usage_category(cpu_usage, memory_usage):
    # Thresholds for defining peak, moderate, and off-peak usage categories
    CPU_PEAK_THRESHOLD = 3.5  # Adjust based on dataset range
    MEMORY_PEAK_THRESHOLD = 0.05  # Adjust based on dataset range
    
    # Determine usage category based on CPU and memory usage
    if cpu_usage > CPU_PEAK_THRESHOLD or memory_usage > MEMORY_PEAK_THRESHOLD:
        return "peak"
    elif cpu_usage > 1.0 * CPU_PEAK_THRESHOLD or memory_usage > 0.035 * MEMORY_PEAK_THRESHOLD:
        return "moderate"
    else:
        return "off-peak"

# Apply usage category and calculate energy consumption for each row
df['usage_category'] = df.apply(lambda row: get_usage_category(row['cpu_usage_distribution'], row['memory_accesses_per_instruction']), axis=1)

# Map usage categories to numeric values: 1 for peak, 2 for moderate, and 3 for off-peak
usage_mapping = {
    'peak': 1,
    'moderate': 2,
    'off-peak': 3
}
df['usage_category_numeric'] = df['usage_category'].map(usage_mapping)

# Ensure energy consumption is calculated correctly for each record
df['energy_consumption'] = df.apply(lambda row: calculate_energy(row['cpu_usage_distribution'], row['memory_accesses_per_instruction'], row['duration']), axis=1)

# Simulate renewable energy generation (Solar and Wind)
def simulate_renewable_energy():
    # Simulate Solar Energy (kW)
    solar_irradiance = random.uniform(100, 1000)  # Solar irradiance between 100 and 1000 W/m^2
    solar_energy = solar_irradiance * 0.1  # Simplified efficiency factor for solar energy

    # Simulate Wind Energy (kW)
    wind_speed = random.uniform(5, 25)  # Wind speed between 5 m/s and 25 m/s
    wind_energy = wind_speed * 0.3  # Simplified efficiency factor for wind energy

    # Total renewable energy contribution (solar + wind)
    total_renewable_energy = solar_energy + wind_energy

    # Cap renewable energy to a reasonable fraction of energy consumption (max 50%)
    return min(total_renewable_energy, 0.5)

# Apply the renewable energy calculation to the dataset
df['renewable_energy'] = df.apply(lambda row: simulate_renewable_energy(), axis=1)

# Ensure that renewable energy factor is within reasonable bounds (percentage)
df['renewable_energy_factor'] = (df['renewable_energy'] / df['energy_consumption']).clip(upper=1) * 100

# Rename and reorder columns for clarity and readability
df.rename(columns={
    'start_time': 'Start Time',
    'end_time': 'End Time',
    'cpu_usage_distribution': 'CPU Usage',
    'memory_accesses_per_instruction': 'Memory Usage',
    'usage_category': 'Usage Scenario',
    'energy_consumption': 'Energy Consumption',
    'duration': 'Duration (Seconds)',
    'renewable_energy': 'Renewable Energy (kW)',
    'renewable_energy_factor': 'Renewable Energy Factor (%)'
}, inplace=True)

# Save the results to a CSV file for further analysis or reporting
output_file = "cloud_simulation_results.csv"
df.to_csv(output_file, index=False)

print(f"Data saved successfully to {output_file}")

# Print the data type of the numeric usage category for verification
print(df['usage_category_numeric'].dtype)

# Summary statistics for energy consumption (descriptive statistics)
energy_consumption_summary = df['Energy Consumption'].describe()
print("Summary Statistics for Energy Consumption (kW):")
print(energy_consumption_summary)
print("\n")

# Summary statistics for renewable energy factor (descriptive statistics)
renewable_energy_factor_summary = df['Renewable Energy Factor (%)'].describe()
print("Summary Statistics for Renewable Energy Factor (%):")
print(renewable_energy_factor_summary)
print("\n")

# Usage scenario distribution (count of each usage category)
usage_distribution = df['Usage Scenario'].value_counts()
print("Usage Scenario Distribution:")
print(usage_distribution)
print("\n")
