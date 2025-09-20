# P4 Machine Learning
# Ernesto Ulises Hernández Martínez
# Not Supervised learning
#           Concrete Compressive Dataset
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

# Concrete Compressive Dataset
concrete = pd.read_excel("../data/concrete+compressive+strength/Concrete_Data.xls")
# despite the dataset having header, is super large, so we will reduce it a little bit
concrete.columns = ['cement', 'blast_furnance_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age', 'concrete_compressive_strength']
#concrete.head()

###################################################################################################
#                   Random Forest Regressor
###################################################################################################
#                   K Neighbors Regressor
###################################################################################################
#                   Linear Regression
###################################################################################################
#                   MLP Regressor
###################################################################################################