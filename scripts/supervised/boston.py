# P4 Machine Learning
# Ernesto Ulises Hernández Martínez
# Not Supervised learning
#           Boston Housing Dataset
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sys
#print(sys.executable)
#print("Current working directory:", os.getcwd())

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# we are going to be using 4 regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Boston Housing Dataset
boston = pd.read_csv("data/boston/housing.csv", header = None, delimiter = r"\s+")
boston.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# the meaning of this columns is going to be explained in the report document
print(boston.head())

X = boston.iloc[:, :-1]  # features
y = boston.iloc[:, -1]   # target

# 70 - 30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

plt.figure(figsize=(12, 8))

###################################################################################################
#                   Random Forest Regressor
RandomForestRegressorModel = RandomForestRegressor() # choose the regressor
RandomForestRegressorModel.fit(X_train, y_train) # train the model
y_pred_rf = RandomForestRegressorModel.predict(X_test) # predict

# Evaluate the model
print("\nRandom Forest Regressor")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R^2 Score:", r2_score(y_test, y_pred_rf))

# Plot the results
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # 45 degree line to visualize performance
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Random Forest")

###################################################################################################
#                   K Neighbors Regressor
KNeighborsRegressorModel = KNeighborsRegressor() # choose the regressor
KNeighborsRegressorModel.fit(X_train, y_train) # train the model
y_pred_knn = KNeighborsRegressorModel.predict(X_test) # predict

# Evaluate the model
print("\nK Neighbors Regressor")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_knn))
print("R^2 Score:", r2_score(y_test, y_pred_knn))

# Plot the results
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_knn, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # 45 degree line to visualize performance
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("KNeighbors")

###################################################################################################
#                   Linear Regression
LinearRegressionModel = LinearRegression() # choose the regressor
LinearRegressionModel.fit(X_train, y_train) # train the model
y_pred_lr = LinearRegressionModel.predict(X_test) # predict

# Evaluate the model
print("\nLinear Regression")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R^2 Score:", r2_score(y_test, y_pred_lr))

# Plot the results
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_lr, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # 45 degree line to visualize performance
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Linear Regression")

###################################################################################################
#                   MLP Regressor
MLPRegressorModel = MLPRegressor()
MLPRegressorModel.fit(X_train, y_train)
y_pred_mlp = MLPRegressorModel.predict(X_test)

# Evaluate the model
print("\nMLP Regressor")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_mlp))
print("R^2 Score:", r2_score(y_test, y_pred_mlp))

# Plot the results
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_mlp, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # 45 degree line to visualize performance
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("MLP Regressor")

###################################################################################################

plt.tight_layout()
plt.savefig("results/boston_supervised_regressors_comparison.png") # Save the figure
plt.show()

# Command to run the script and save the terminal output
#python scripts/supervised/boston.py | tee results/boston_supervised_terminal_output.txt