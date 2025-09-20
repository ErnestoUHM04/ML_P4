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

# we are going to be using 4 regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Iris Dataset
iris = pd.read_csv("../data/iris/iris.data", header = None)
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
#iris.head() # check it
#iris.info()
#iris.describe()

###################################################################################################
#                   Random Forest Regressor

###################################################################################################
#                   K Neighbors Regressor
###################################################################################################
#                   Linear Regression
###################################################################################################
#                   MLP Regressor
###################################################################################################