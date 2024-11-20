# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:43:53 2024

@author: Krist
"""
import numpy as np
import random
import os

os.chdir("C:/Users/Krist/OneDrive/Dokumenter/Bachelorprojekt/bachelor_projekt/Functions")

from functions import Cell_terrain_summery_matrix
from functions import Gaussian_Density
from functions import LIDAR_LOADER
from functions import AdjacencyAndTransmission_matrix
from functions import PlotStuff
from functions import Cell_Z_summery_matrix


#setting the seed:
random.seed(12345)
    
    
import matplotlib.pyplot as plt

#file
file="C:/Users/Krist\OneDrive/Dokumenter/Bachelorprojekt/GIS files/LHD_FXX_0772_6282_PTS_O_LAMB93_IGN69.copc.laz"

#we only load every 100 points
X,Y,Z,XY_class = LIDAR_LOADER(file, 100)

sample_indexes= np.random.choice(np.size(X),100)
X_test = X[sample_indexes]
Y_test = Y[sample_indexes]
Critical_Distance=100

#in this test, we observe how changing the the size L, while keeping all other parameters fixed.

L=5
L5_M_cell_summery = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L)
fig,ax=plt.subplots()
ax.set(xlim=(0,L-1),ylim=(L-1,0))
ax.matshow(L5_M_cell_summery)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
print("done with L=5")




L=10
L10_M_cell_summery = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L)
fig,ax=plt.subplots()
ax.set(xlim=(0,L-1),ylim=(L-1,0))
ax.matshow(L10_M_cell_summery)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

print("done with L=10")


L=25
L25_M_cell_summery = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L)
fig,ax=plt.subplots()
ax.set(xlim=(0,L-1),ylim=(L-1,0))
ax.matshow(L25_M_cell_summery)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
print("done with L=25")


L=50
L50_M_cell_summery = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L)
fig,ax=plt.subplots()
ax.set(xlim=(0,L-1),ylim=(L-1,0))
ax.matshow(L50_M_cell_summery)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
print("done with L=50")

L=100
L100_M_cell_summery = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L)
fig,ax=plt.subplots()
ax.set(xlim=(0,L-1),ylim=(L-1,0))
ax.matshow(L100_M_cell_summery)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
print("done with L=100")

L=200
L200_M_cell_summery = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L)
fig,ax=plt.subplots()
ax.set(xlim=(0,L-1),ylim=(L-1,0))
ax.matshow(L200_M_cell_summery)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
print("done with L=200")

L=400
L400_M_cell_summery = Cell_terrain_summery_matrix(X,Y,Z,XY_class,L)
fig,ax=plt.subplots()
ax.set(xlim=(0,L-1),ylim=(L-1,0))
ax.matshow(L400_M_cell_summery)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
print("done with L=400")













