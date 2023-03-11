# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 18:16:40 2023

@author: deseyi
"""

import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import math
from sklearn.model_selection import train_test_split
import numpy as np


def haversine(x1, x2):
    """
    Calculate the Haversine distance between two points specified by 
    their latitude and longitude in decimal degrees.
    """
   
    lat1, lon1 = x1[0], x1[1]
    lat2, lon2 = x2[0], x2[1]
   
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


#Setting working directory
path = 'C:\\Users\\desey\\OneDrive\\Desktop\\MACHINE_LEARN\\Project_2'
os.chdir(path)

#Reading the dataset
a = pd.read_csv('Preds.csv')
b = a.copy()



#Extracting certain columns to be used as X variables
cols = [1,2]
X = a[a.columns[cols]]
X

Y = b['predLR']
Y2 = b['pred_NB']


# Load your data into a pandas dataframe, including columns for latitude and longitude.
# Define your input features and target variable(s).  

# Split into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=99)

# Create a KNN model using haversine distance
knn = KNeighborsRegressor(n_neighbors=5, metric=haversine, weights= 'distance')
knn.fit(X_train, y_train)


#For KNN Model 1
ypred1 = knn.predict(X_test)
plt.plot(y_test, ypred1, 'ro')
plt.xlabel('Risk Obtained from LR Model' )
plt.ylabel('Risk Predicted by KNN Model' )
plt.title('Obs VS Pred for Testing Data')
plt.grid()
plt.show()

# Split into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,Y2,test_size=0.25,random_state=99)


knn2 = KNeighborsRegressor(n_neighbors=5, metric=haversine, weights= 'distance')
knn2.fit(X_train, y_train)


#For KNN Model 1
ypred1 = knn.predict(X_test)
plt.plot(y_test, ypred1, 'ro')
plt.xlabel('Risk Obtained from NB Model' )
plt.ylabel('Risk Predicted by KNN Model' )
plt.title('Obs VS Pred for Testing Data')
plt.grid()
plt.show()

#MInimum and Maximum Latitude Extracted from QGIS
minlat = 30.6658624138208680
maxlat = 36.5012320444111253

minlon = -103.6083311841115346
maxlon = -100.4667664246455843

step = 0.01 #1degree = 111 km (or 60 nautical miles)
lat_all = np.arange(minlat, maxlat+step, step)
lon_all = np.arange(minlon, maxlon+step, step)



yv, xv = np.meshgrid(lon_all, lat_all)
all_points = pd.DataFrame(dict(x=xv.ravel(), y=yv.ravel()))
all_points = all_points.rename(columns={'x':'LatDD', 'y':'LonDD'})

final_df = all_points.copy()

res = knn.predict(all_points)
res

final_df['LR_res'] = res

res2 = knn2.predict(all_points)
res2

final_df['NBC_res'] = res2

#Weighing each method
w1 = 0.65 #Weightage for LR
w2 = 1 - w1 #Weightage for NB
res_final = w1 * res + w2 * res2

#Final Vulnerability Map
final_df['res_final'] = w1 * final_df['LR_res'] + w2 * final_df['NBC_res']
final_df



final_df.to_csv('Final_Risk_Val.csv')
