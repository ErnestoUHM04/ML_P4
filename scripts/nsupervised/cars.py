# P4 Machine Learning
# Ernesto Ulises Hernández Martínez
# Not Supervised learning
#           Car Price Prediction Dataset
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sys
#print(sys.executable)

# we are going to be using 4 regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cluster import KMeans # For K-Means clustering
from sklearn.preprocessing import LabelEncoder

# Car Price Prediction Dataset
cars = pd.read_csv("data/cars/car_prediction_data.csv")
print(cars.head())

le = LabelEncoder()
cars['Car_Name'] = le.fit_transform(cars['Car_Name'])
cars['Fuel_Type'] = le.fit_transform(cars['Fuel_Type'])
cars['Seller_Type'] = le.fit_transform(cars['Seller_Type'])
cars['Transmission'] = le.fit_transform(cars['Transmission'])

X = cars[['Car_Name','Year','Selling_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
#y = cars[['Present_Price']] # since this is non supervised learning, we are not going to be using it

# with the optimal k script turns out the best k is k = 2
# we are also trying then when k = 10
k = 4

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)

# this is a 70 - 30 split
# Use clusters as pseudo-targets
X_train, X_test, y_train, y_test = train_test_split(X, clusters, test_size=0.3, random_state=42)

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
#plt.savefig("results/cars_nsupervised_k2_regressors_comparison.png") # Save the figure
plt.savefig("results/cars_nsupervised_k4_regressors_comparison.png") # Save the figure
#plt.savefig("results/cars_nsupervised_k10_regressors_comparison.png") # Save the figure
plt.show()

# Command to run the script and save the terminal output
#python scripts/nsupervised/cars.py | tee results/cars_nsupervised_k2_terminal_output.txt
#python scripts/nsupervised/cars.py | tee results/cars_nsupervised_k4_terminal_output.txt
#python scripts/nsupervised/cars.py | tee results/cars_nsupervised_k10_terminal_output.txt