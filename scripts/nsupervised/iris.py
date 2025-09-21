# P4 Machine Learning
# Ernesto Ulises Hernández Martínez
# Not Supervised learning
#           Iris Dataset
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sys
#print(sys.executable)
#print("Current working directory:", os.getcwd())

# we are going to be using 4 regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Iris Dataset
iris = pd.read_csv("data/iris/iris.data", header = None)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(iris.head()) # check it
#iris.info()
#iris.describe()

X = iris.iloc[:, :-1]  # features
#y = iris.iloc[:, -1]   # target.  We are not going to use it

# this is a 80 - 20 split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Because we are going to work with not supervised learning, we are going to predict and not use the last column
###################################################################################################
#                   Random Forest Regressor
RandomForestRegressorModel = RandomForestRegressor()
RandomForestRegressorModel.fit(X_train)

###################################################################################################
#                   K Neighbors Regressor
KNeighborsRegressorModel = KNeighborsRegressor()
KNeighborsRegressorModel.fit(X_train)

###################################################################################################
#                   Linear Regression
LinearRegressionModel = LinearRegression()
LinearRegressionModel.fit(X_train)

###################################################################################################
#                   MLP Regressor
MLPRegressorModel = MLPRegressor()
MLPRegressorModel.fit(X_train)

###################################################################################################