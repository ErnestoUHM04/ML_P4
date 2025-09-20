# P4 Machine Learning
# Ernesto Ulises Hernández Martínez
# Not Supervised learning
#           Diabetes Dataset
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

# Diabetes Dataset
diabetes = pd.read_csv("../data/diabetes/diabetes.csv")
#diabetes.head()

###################################################################################################
#                   Random Forest Regressor
###################################################################################################
#                   K Neighbors Regressor
###################################################################################################
#                   Linear Regression
###################################################################################################
#                   MLP Regressor
###################################################################################################