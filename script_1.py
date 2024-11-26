                                                                 #Preprosessing data
import subprocess
import sys

# Function to check and install missing libraries
def install_libraries():
    required_libraries = ['pandas', 'numpy', 'ast', 're']
    
    for library in required_libraries:
        try:
            # Check if the library is installed
            __import__(library)
        except ImportError:
            # If not, install it
            print(f"{library} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])

# Installing the libraries
install_libraries()

# importing the libraries after installation
import pandas as pd
import numpy as np
import ast
import re

# loading the dataset
df = pd.read_csv("dataset.csv")

# cleaning the data
df.drop_duplicates(inplace=True)  # removing any duplicate rows
df['memory_accesses_per_instruction'] = df['memory_accesses_per_instruction'].fillna(df['memory_accesses_per_instruction'].mean())  # filling missing values with the column mean

# selecting features to keep
columns_to_keep = [
    'cpu_usage_distribution',
    'memory_accesses_per_instruction',
    'instance_events_type',
    'scheduling_class',
    'priority',
    'time',
    'start_time',
    'end_time',
    'machine_id',
    'cluster',
    'failed'
]
df = df[columns_to_keep]  # keeping only the selected columns

# cleaning and converting list-like strings into float values
def clean_cpu_usage(value):
    if isinstance(value, str):
        try:
            # removing unwanted characters and extra spaces
            value = value.replace("\n", "").replace(" ", "").strip()

            # handling missing commas between numbers
            value = re.sub(r'(\d)(?=\d*\.\d)', r'\1,', value)

            # fixing cases where scientific notation is used without proper separation
            value = re.sub(r'(?<=\d)(?=\d*e)', r',', value)

            # converting string to a list
            if value.startswith('[') and value.endswith(']'):
                value = ast.literal_eval(value)
                if isinstance(value, list) and len(value) > 0:
                    # calculating and returning the mean of the list as a float
                    return float(np.mean(np.array(value)))
            else:
                return np.nan  # returning NaN if the value isn't properly formatted
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing value: {value} | Error: {e}")
            return np.nan  # returning NaN in case of an error
    return np.nan  # returning NaN for non-string values

# applying the cleaning function to the column
df['cpu_usage_distribution'] = df['cpu_usage_distribution'].apply(clean_cpu_usage)

# checking the cleaned data
print(df['cpu_usage_distribution'].head(20))

# replacing NaN values with the column mean
df['cpu_usage_distribution'] = df['cpu_usage_distribution'].fillna(df['cpu_usage_distribution'].mean())

# converting the column to numeric type and ensuring no NaN values are left
df['cpu_usage_distribution'] = pd.to_numeric(df['cpu_usage_distribution'], errors='coerce')
df['cpu_usage_distribution'] = df['cpu_usage_distribution'].astype('float32')

# checking again to make sure there are no NaNs left
print(df['cpu_usage_distribution'].head(20))

# feature engineering for 'estimated_power_consumption' using an improved formula

# constants (you can tweak these values based on your system characteristics)
cpu_weight = 0.05  # CPU usage weight (adjustable based on empirical data)
memory_weight = 0.02  # memory accesses weight (adjustable)

# calculating power consumption based on CPU and memory usage
df['estimated_power_consumption'] = (df['cpu_usage_distribution'] * cpu_weight) + (df['memory_accesses_per_instruction'] * memory_weight)

# checking the result
print(df[['cpu_usage_distribution', 'memory_accesses_per_instruction', 'estimated_power_consumption']].head(20))

# saving the cleaned dataset with the new power consumption estimate
df.to_csv("cleaned_dataset.csv", index=False)
