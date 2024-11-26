                                                                       #Comparing Models
import subprocess
import sys

# Function to check and install missing libraries
def install_libraries():
    # List of required libraries
    required_libraries = ['pandas', 'numpy', 'xgboost', 'sklearn', 'deap']
    
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
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from deap import base, creator, tools, algorithms

# Load the dataset
df = pd.read_csv("cleaned_dataset.csv")

"""

# Data Cleaning
df.drop_duplicates(inplace=True)
df['memory_accesses_per_instruction'].fillna(df['memory_accesses_per_instruction'].mean(), inplace=True)

# Feature Selection
columns_to_keep = ['instance_events_type', 'memory_accesses_per_instruction', 'failed', 'time']
df = df[columns_to_keep]

# Feature Engineering
df['estimated_power_consumption'] = df['memory_accesses_per_instruction'] * 100  # Example calculation

"""

# Set the target column and target variable
X = df.drop('estimated_power_consumption', axis=1)
y = df['estimated_power_consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### 1. XGBoost Model ###
model_xgb = xgb.XGBRegressor()
model_xgb.fit(X_train_scaled, y_train)
y_pred_xgb = model_xgb.predict(X_test_scaled)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"Mean Squared Error of XGBoost: {mse_xgb}")

### 2. DQN approximation using MLP Regressor ###
model_dqn = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
model_dqn.fit(X_train_scaled, y_train)
y_pred_dqn = model_dqn.predict(X_test_scaled)
mse_dqn = mean_squared_error(y_test, y_pred_dqn)
print(f"Mean Squared Error of DQN Approximation: {mse_dqn}")

### 3. Genetic Algorithm (GA) Optimization ###

# DEAP GA Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(X_train_scaled[0]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_individual(individual):
    # Use individual as model weights for a simple linear model
    y_pred = np.dot(X_train_scaled, np.array(individual))
    return (mean_squared_error(y_train, y_pred),)

toolbox.register("evaluate", eval_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# GA Hyperparameters
pop = toolbox.population(n=50)
hof = tools.HallOfFame(1)

algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, halloffame=hof, verbose=False)

# Use the best individual from GA
best_individual = hof[0]
y_pred_ga = np.dot(X_test_scaled, np.array(best_individual))
mse_ga = mean_squared_error(y_test, y_pred_ga)
print(f"Mean Squared Error of GA Optimized Model: {mse_ga}")

import matplotlib.pyplot as plt

# Plot Actual vs Predicted for all models
plt.figure(figsize=(12, 6))

# XGBoost
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_xgb, color='blue', label='XGBoost', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('XGBoost: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# DQN MLP
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_dqn, color='green', label='DQN MLP', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('DQN MLP: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Genetic Algorithm
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_ga, color='purple', label='GA', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Genetic Algorithm: Actual vs Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.savefig('prediction_comparison.png') # Save the image
plt.show()

# Bar chart comparing MSE
mse_values = [mse_xgb, mse_dqn, mse_ga]  # MSE values for XGBoost, DQN MLP, and GA
models = ['XGBoost', 'DQN MLP', 'GA']

plt.figure(figsize=(8, 5))
plt.bar(models, mse_values, color=['blue', 'green', 'purple'])
plt.title('MSE Comparison of Models')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.savefig('mse_comparison.png') # Save the image
plt.show()
