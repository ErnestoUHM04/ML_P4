# P4 Machine Learning
# Ernesto Ulises Hernández Martínez
# Not Supervised learning
#           Wine Quality Dataset
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

# Wine Quality Dataset
# there is red and white wine data
red = pd.read_csv("../data/wine+quality/winequality-red.csv", sep = ';')
#red.head()
white = pd.read_csv("../data/wine+quality/winequality-white.csv", sep = ';')
#white.head()
red['type'] = 'red'
white['type'] = 'white'

wine = pd.concat([red, white], ignore_index = True)
#white.head()
# Check
#print(wine.head())
#print(wine['type'].value_counts())

###################################################################################################
#                   Random Forest Regressor
###################################################################################################
#                   K Neighbors Regressor
###################################################################################################
#                   Linear Regression
###################################################################################################
#                   MLP Regressor
###################################################################################################