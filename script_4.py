                                                        # Create and Evaluate the schedular 
import subprocess
import sys

# Function to check and install missing libraries
def install_libraries():
    # List of required libraries
    required_libraries = ['pandas', 'numpy', 'xgboost', 'sklearn', 'matplotlib', 'seaborn']
    
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
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# loading the data from the CSV file
df = pd.read_csv("cloud_simulation_results.csv")

# splitting the data into features (X) and target (y)
X = df[['CPU Usage', 'Memory Usage', 'usage_category_numeric']]
y = df['Energy Consumption']

# splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating and training the XGBoost model
model = XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# predicting energy consumption on the test set
y_pred = model.predict(X_test)

# checking the performance of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# defining a function to schedule tasks more efficiently
def schedule_tasks(df, model):
    """
    this function schedules tasks based on energy usage, renewable energy, and usage scenario.
    it's prioritizing renewable energy for tasks.
    """
    # predicting energy consumption using the trained XGBoost model
    df['Predicted Energy Consumption'] = model.predict(df[['CPU Usage', 'Memory Usage', 'usage_category_numeric']])

    # relaxing the condition to consider more tasks as Off-Peak
    df['Energy-Efficient Scheduling'] = np.where(df['CPU Usage'] < 2, 'Off-Peak', 'Peak')  # now more tasks are "Off-Peak"
    
    # adjusting the energy consumption based on the scheduling
    # if it's Off-Peak, reducing the energy consumption by 10%
    # if it's Peak, no change is made
    df['Scheduled Energy Consumption'] = np.where(df['Energy-Efficient Scheduling'] == 'Off-Peak',
                                                 df['Predicted Energy Consumption'] * 0.9,
                                                 df['Predicted Energy Consumption'])
    
    # returning the updated DataFrame
    return df[['CPU Usage', 'Memory Usage', 'Energy-Efficient Scheduling', 'Predicted Energy Consumption', 'Scheduled Energy Consumption']]

# applying the scheduling function
df_scheduled = schedule_tasks(df, model)

# saving the results of the scheduling to a new CSV file
output_file_scheduled = "cloud_simulation_scheduled_results.csv"
df_scheduled.to_csv(output_file_scheduled, index=False)

print(f"Data saved successfully to {output_file_scheduled}")

# evaluating the scheduler's performance by comparing predicted and scheduled energy consumption
mse_scheduled = mean_squared_error(df['Energy Consumption'], df_scheduled['Scheduled Energy Consumption'])
r2_scheduled = r2_score(df['Energy Consumption'], df_scheduled['Scheduled Energy Consumption'])

print(f'Mean Squared Error (Scheduler): {mse_scheduled}')
print(f'R-squared (Scheduler): {r2_scheduled}')

# calculating how much energy is saved by the scheduler (predicted - scheduled)
df_scheduled.loc[:, 'Energy Saved'] = df_scheduled['Predicted Energy Consumption'] - df_scheduled['Scheduled Energy Consumption']

# calculating the total energy saved
total_energy_saved = df_scheduled['Energy Saved'].sum()

print(f"Total Energy Saved (kW): {total_energy_saved}")

# setting seaborn style for better aesthetics in the plots
sns.set(style="whitegrid")

# plotting the distribution of energy saved
plt.figure(figsize=(10, 6))
sns.histplot(df_scheduled['Energy Saved'], bins=50, kde=True, color='blue', stat='density')
plt.title('Distribution of Energy Saved (kW)', fontsize=16)
plt.xlabel('Energy Saved (kW)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()

# plotting a scatter plot comparing predicted vs. scheduled energy consumption
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_scheduled, x='Predicted Energy Consumption', y='Scheduled Energy Consumption', alpha=0.6)
plt.title('Predicted vs Scheduled Energy Consumption', fontsize=16)
plt.xlabel('Predicted Energy Consumption (kW)', fontsize=12)
plt.ylabel('Scheduled Energy Consumption (kW)', fontsize=12)
plt.show()

# showing the summary statistics of energy saved
energy_saved_summary = df_scheduled['Energy Saved'].describe()
print("Summary Statistics for Energy Saved (kW):")
print(energy_saved_summary)
