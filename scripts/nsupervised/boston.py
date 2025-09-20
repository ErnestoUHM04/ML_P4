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

# we are going to be using 4 regressors
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Boston Housing Dataset
boston = pd.read_csv("../data/boston/housing.csv", header = None, delimiter = r"\s+")
boston.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
# the meaning of this columns is going to be explained in the report document
#boston.head() 

###################################################################################################
#                   Random Forest Regressor
###################################################################################################
#                   K Neighbors Regressor
###################################################################################################
#                   Linear Regression
###################################################################################################
#                   MLP Regressor
###################################################################################################