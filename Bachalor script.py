# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:10:15 2024

@author: Kristian Juul Rasmussen
"""

import numpy as np
import random
import os

os.chdir("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Functions")

from functions_test import Cell_terrain_summery_matrix
from functions_test import Gaussian_Density
from functions_test import LIDAR_LOADER
from functions_test import AdjacencyAndTransmission_matrix
from functions_test import PlotStuff
from functions_test import Mosquito_Human_Count
from functions_test import Produce_Grid_Coordinates
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

#setting the seed:
random.seed(12345)
    
    
import matplotlib.pyplot as plt

 


######## 1: Load in the lidar data

#file
file="C:/Users/Krist\OneDrive/Dokumenter/Bachelorprojekt/GIS files/LHD_FXX_0772_6282_PTS_O_LAMB93_IGN69.copc.laz"

#we only load every 100 points
X,Y,Z,XY_class = LIDAR_LOADER(file, 100)

print("LOADED LIDAR")

print(np.unique(XY_class))



#we set a critical distance, which represents the maximum flying distance that can connect two nodes.  
L=200
Critical_Distance=100
n_mosquitos = 500000
n_people= 4000


#from these points in space, we draw n number of them, and use their coordiantes as our nodes.


###### random coordinates #####
#n_nodes=300
#sample_indexes= np.random.choice(np.size(X),n_nodes)
#X_nodes = X[sample_indexes]
#Y_nodes = Y[sample_indexes]

#####  grid coodinates #####

X_nodes,Y_nodes=Produce_Grid_Coordinates(X,Y,L,skip_n=10)

plt.scatter(X_nodes,Y_nodes)
#plt.scatter(X_nodes,Y_nodes,s=0.1)





##### 2:  Devide the 1KM by 1KM space into cells, and produce an evaluation for each cell.
#           returning a score explaining how "good" the conditions are for mosquitos. 





#using LIDAR designations, and 


M_mosquito, M_Human, MM_Unclassified, MM_Ground,MM_Low_Vegitation,MM_Medium_vegitation,MM_High_vegitation,MM_Buildings,MM_Water,MM_Bridge_Decks,MM_64,MM_65,MM_66 = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L,n_people,n_mosquitos)




#we make a ternary predicter

import pandas as pd
# Example dataset
data = pd.DataFrame({
  #Site A, Site B, Site C, Site D, Site E, Site F, Site G, Site H, Site H, Site I, Site J, site G, corners
  
    'vegetation':   [68.97, 66.20, 60.20, 88.64, 96.35 ,81.71,  47.875, 24.77, 55.10, 17.38, 100., 0.  , 0.], #pecentage vegitation
    'water':        [7.58 , 9.02 , 2.93 , 4.25 , 0.00  ,4.78 ,  36.336, 62.14, 31.92, 76.22, 0.  , 100., 0.],        # Percentage of water
    'buildings':    [23.45, 24.78, 36.87, 7.11 , 3.65  ,13.51,  15.789, 13.09, 12.98, 6.40 , 0.  , 0.   , 100.],    # Percentage of buildings
    'mosquitos':    [56,       44,   106,   22 , 10    ,69   ,  377  ,   177, 35   , 232   , 1.  , 1.  , 1.]  # Mosquito count
    
})


# Step 1: Create the ternary grid
def create_ternary_grid(n=50):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    valid = (Z >= 0) & (Z <= 1)
    return X[valid], Y[valid], Z[valid]

X, Y, Z = create_ternary_grid()

# Step 2: Interpolate mosquito counts
points = data[['water', 'vegetation']].values / 100
values = data['mosquitos'].values
grid_mosquitos = griddata(points, values, (X, Y), method='cubic')

# Step 3: Create prediction function
def predict_mosquitos(water, vegetation, building):
    if abs(water + vegetation + building - 100) > 1e-6:
        raise ValueError("Percentages must sum to 100")
    
    water, vegetation = water / 100, vegetation / 100
    predicted = griddata(points, values, (water, vegetation), method='cubic')
    return float(predicted)





M_Vegitation = (
    
    (1/2)*MM_Ground +
    MM_Low_Vegitation+
    MM_Medium_vegitation+
    MM_High_vegitation+
    
    (1/3)*MM_64+
    (1/3)*MM_65+
    (0/10)*MM_Unclassified
    
    
    )*100


M_Manmade_Structures = (
    
    (1/2)*MM_Ground +
    MM_Buildings+
    MM_Bridge_Decks+
    
    
    (1/3)*MM_64+
    (1/3)*MM_65+
    (10/10)*MM_66+  #various buildings
    (2/10)*MM_Unclassified
    
    
    )*100


M_Water = (
    MM_Water+
    
    (1/3)*MM_64+
    (1/3)*MM_65+
    (8/10)*MM_Unclassified
    
    
    
    )*100



# Create an empty array to store the predictions
predicted_mosquitos = np.zeros((L, L))

# Vectorize the predict_mosquitos function for efficiency
vectorized_predict = np.vectorize(predict_mosquitos)

# Apply the prediction function to all elements
predicted_mosquitos = vectorized_predict(M_Water, M_Vegitation, M_Manmade_Structures)

# Example: Print the prediction for a specific location
print(f"Predicted mosquitos at [10, 10]: {predicted_mosquitos[10, 10]}")


M_mosquito=predicted_mosquitos

sigma=1

M_mosquito = gaussian_filter(M_mosquito,sigma)

#we apply some simple corrections, making sure none of the elements are less than 0

M_mosquito[M_mosquito<0]=0

#now... considering the fact the matrice is most likley going to sum to sometihng in the multi millions...
#that is ridicilous in most cases. We rescale the matrix to have so as to have the desired sum n_mosquitos


#normalize the dataset
M_mosquito = (M_mosquito - np.min(M_mosquito)) / (np.max(M_mosquito) - np.min(M_mosquito))
#find the find the sum scale 
M_mosquito_sum_before=sum([np.sum(M_mosquito) for i in M_mosquito])
M_mosquito_sum_desired = n_mosquitos
Mosquito_scale_fraction = M_mosquito_sum_desired / M_mosquito_sum_before
#scaling approapriatly
M_mosquito = M_mosquito * Mosquito_scale_fraction





plt.matshow(M_mosquito)
plt.matshow(M_Human)



print("Produced M_Cell_Summery")




#now we have an inerperation of the distribution of peple andmmosquitos.
#now we need an "itneraction" interpetation between two cells



##### 3:  we are now going to produce a series of matrices. 

#for this, we set L to 200. deviding a so each cell is 5x5 square




#Gaussian_Mosquito_density_M,Gaussian_Human_density_M = Gaussian_Density(X_nodes,Y_nodes,L, M_mosquito,M_Human)



#print("Produced Gaussian_Density_M")


Mosquito_Count,Human_Count = Mosquito_Human_Count(X_nodes,Y_nodes,L, M_mosquito, M_Human)
Mosquito_Count=Mosquito_Count.reshape((np.size(X_nodes),))
Human_Count=Human_Count.reshape((np.size(X_nodes),))
Proximity_M, Transmission_M = AdjacencyAndTransmission_matrix (X_nodes,Y_nodes,L,Critical_Distance)
XY_info=np.asarray([X_nodes,Y_nodes,Mosquito_Count,Human_Count])


plt.matshow(Proximity_M)
plt.matshow(Transmission_M)


print("Produced Proximity_M")
print("Produced Transmission_M")

plt.matshow(Proximity_M)
plt.matshow(Transmission_M)

os.chdir("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Matrices saved")

#removing them in case they already exist
os.remove("XY_info.csv")
#os.remove("Gaussian_Mosquito_density_M.csv")
#os.remove("Gaussian_Human_density_M.csv")

os.remove("M_mosquito.csv")
os.remove("M_Human.csv")

os.remove("Proximity_M.csv")
os.remove("Transmission_M.csv")



#varrible list= [Critical_Distance, L, n_people, n_nodes]



#save them as pandas arrays
import pandas as pd
XY_info = pd.DataFrame(XY_info)

#Gaussian_Mosquito_density_M=pd.DataFrame(Gaussian_Mosquito_density_M)
#Gaussian_Human_density_M=pd.DataFrame(Gaussian_Human_density_M)


M_mosquito=pd.DataFrame(M_mosquito)
M_Human=pd.DataFrame(M_Human)
Proximity_M=pd.DataFrame(Proximity_M)
Transmission_M=pd.DataFrame(Transmission_M)


XY_info.to_csv('XY_info.csv', index=False, header=False)
#Gaussian_Mosquito_density_M.to_csv('Gaussian_Mosquito_density_M.csv', index=False, header=False)
#Gaussian_Human_density_M.to_csv('Gaussian_Human_density_M.csv', index=False, header=False)

M_mosquito.to_csv('M_mosquito.csv', index=False, header=False)
M_Human.to_csv('M_Human.csv', index=False, header=False)


Proximity_M.to_csv('Proximity_M.csv', index=False, header=False)
Transmission_M.to_csv('Transmission_M.csv', index=False, header=False)










 




plt.matshow(Transmission_M)














