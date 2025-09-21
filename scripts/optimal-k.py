# -*- coding: utf-8 -*-
"""
Created on

@author: M
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import sys
import os

from sklearn.preprocessing import LabelEncoder

print("Current working directory:", os.getcwd())


# sys.stdout = open('resultado_clustering.txt', 'w')
class Logger(object):
    def __init__(self):
        self.terminal = sys.__stdout__
        #self.log = open("results/boston_resultado_clustering.txt", "w", encoding="utf-8")
        #self.log = open("results/iris_resultado_clustering.txt", "w", encoding="utf-8")
        #self.log = open("results/diabetes_resultado_clustering.txt", "w", encoding="utf-8")
        #self.log = open("results/wine_resultado_clustering.txt", "w", encoding="utf-8")
        #self.log = open("results/cars_resultado_clustering.txt", "w", encoding="utf-8")
        self.log = open("results/concrete_resultado_clustering.txt", "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

#dataframe = pd.read_csv('Mall_Customers.csv')
#dataframe = pd.read_csv('Clusters_spi_global_rankings.csv', encoding = 'iso-8859-1')
#dataframe = pd.read_csv('data/iris/iris.data', header = None)
#dataframe = pd.read_csv('data/diabetes/diabetes.csv')
#red = pd.read_csv("data/wine+quality/winequality-red.csv", sep = ';')
#red.head()
#white = pd.read_csv("data/wine+quality/winequality-white.csv", sep = ';')
#white.head()
#red['type'] = 'red'
#white['type'] = 'white'

#dataframe = pd.concat([red, white], ignore_index = True)
#dataframe = pd.read_csv("data/cars/car_prediction_data.csv")
dataframe = pd.read_excel("data/concrete+compressive+strength/Concrete_Data.xls")

# Encode type as numbers
#le = LabelEncoder()
#dataframe['type'] = le.fit_transform(dataframe['type'])
le = LabelEncoder()
#dataframe['Car_Name'] = le.fit_transform(dataframe['Car_Name'])
#dataframe['Fuel_Type'] = le.fit_transform(dataframe['Fuel_Type'])
#dataframe['Seller_Type'] = le.fit_transform(dataframe['Seller_Type'])
#dataframe['Transmission'] = le.fit_transform(dataframe['Transmission'])


#dataframe.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
#dataframe = pd.read_csv('data/boston/housing.csv', header = None, delimiter = r"\s+")
#dataframe.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataframe.columns = ['cement', 'blast_furnance_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age', 'concrete_compressive_strength']

# prueba
# encoding = 'utf-8'
# encoding = 'latin1'
# encoding = 'iso-8859-1'
# encoding = 'cp1252'
#dataframe.columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality","type"]
#Obtiene las caracteristicas de interés
#X = dataframe[['Age','Annual Income (k$)','Spending Score (1-100)']]
#X = dataframe[['name','league','off', 'def']]
#X = dataframe[['off', 'def']]
#X = dataframe.iloc[:, :-1]  # features
#X = dataframe[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","type"]]  # exclude the quality

#X = dataframe[['Car_Name','Year','Selling_Price','Kms_Driven','Fuel_Type','Seller_Type','Transmission','Owner']]
X = dataframe.iloc[:, :-1]
# Silueta

best_sil = 0
best_k = 0

#almacena las métricas
calinski = []
davies = []
sils =[]
n_clusters = []

best_cal = 0
cal_k = 0

best_dav = float('inf')
dav_k = 0


for k in range(2,11):
    plt.figure()
    modelo_prueba = KMeans(n_clusters=k, random_state=42,n_init=10)
    silhouette_visualizer = SilhouetteVisualizer(modelo_prueba, colors='yellowbrick') 
    silhouette_visualizer.fit(X)
    
    score = silhouette_visualizer.silhouette_score_
    calinski_score = calinski_harabasz_score(X, modelo_prueba.labels_)
    calinski.append(calinski_score)
    davies_score = davies_bouldin_score(X, modelo_prueba.labels_)   
    davies.append(davies_score) 
   
    n_clusters.append(k)
    
    if score > best_sil:
       best_sil = score
       best_k = k
    
    if calinski_score > best_cal:
       best_cal = calinski_score
       cal_k = k
       
    if davies_score < best_dav:
       best_dav = davies_score
       dav_k = k   
       
    print(f"Coeficiente de silueta con {k} clusters : {score:.2f}")
    silhouette_visualizer.show()
 

print(f"\nRESULTADO: El mejor coeficiente de silueta fue {best_sil:.2f} con {best_k} clusters\n")


plt.figure()
plt.plot(n_clusters, calinski, marker='o')
plt.title('Índice de Calinski-Harabasz para diferentes valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Calinski-Harabasz Score')
plt.scatter(cal_k, best_cal, color='red', label=f'Máximo ({cal_k}, {best_cal})', zorder=5)
plt.grid(True)
plt.show()
print(f"\nEl mejor valor para k de acuerdo con Calinski-Harabasz es: {cal_k} con un score de {best_cal:.2f}\n")

plt.figure()
plt.plot(n_clusters, davies, marker='o')
plt.title('Índice de Davies-Bouldin para diferentes valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Davies-Bouldin Score')
plt.scatter(dav_k, best_dav, color='red', label=f'Mínimo ({dav_k}, {best_dav})', zorder=5)
plt.grid(True)
plt.show()
print(f"\nEl mejor valor para k de acuerdo con Davies-Bouldin es: {dav_k} con un score de {best_dav:.2f}\n")

plt.figure()
elbow_visualizer = KElbowVisualizer(modelo_prueba, k=(2, 11))
elbow_visualizer.fit(X)
elbow_visualizer.show()
plt.show()
best_k = elbow_visualizer.elbow_value_
print(f"\nEl mejor valor para k de acuerdo con elbow es: {best_k}")


#sys.stdout.close()