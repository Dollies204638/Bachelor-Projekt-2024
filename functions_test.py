# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 13:47:36 2024

@author: Kristian Juul Rasmussen
"""

import numpy as np
import random 
from scipy.interpolate import griddata

def Produce_Grid_Coordinates(X,Y,L,skip_n=1):
    
    
    
    # Calculate how many nodes we will use based on L and skip_n
    L_used = int(L / skip_n)
    
    # Create grid coordinates with proper spacing and offset
    x_coords = np.arange(0, L_used) * skip_n + 0.5
    y_coords = np.arange(0, L_used) * skip_n + 0.5
    
    # Generate the full grid using meshgrid
    X_nodes, Y_nodes = np.meshgrid(x_coords, y_coords)


    X_nodes_flattened=X_nodes.flatten()
    Y_nodes_flattened=Y_nodes.flatten()

    #rescaling
    
    #noting down the vals 
    X_min=np.min(X)
    X_max=np.max(X)
    
    Y_min=np.min(Y)
    Y_max=np.max(Y)
         
    X_nodes_min = np.min(X_nodes_flattened)
    X_nodes_max = np.max(X_nodes_flattened)
    
    Y_nodes_min = np.min(Y_nodes_flattened)
    Y_nodes_max = np.max(Y_nodes_flattened)
    
    

    #scaling the vals.

    OldRange_X = (X_nodes_max - X_nodes_min)  
    NewRange_X = (X_max - X_min)  
    
    X_nodes_scaled = (((X_nodes_flattened - X_nodes_min) * NewRange_X) / OldRange_X) + X_min
    
    
    OldRange_Y = (Y_nodes_max - Y_nodes_min)  
    NewRange_Y = (Y_max - Y_min)  
    
    Y_nodes_scaled = (((Y_nodes_flattened - Y_nodes_min) * NewRange_Y) / OldRange_Y) + Y_min
    

    
    
    
    
    return(X_nodes_scaled,Y_nodes_scaled)
    
    
    
    


def LIDAR_LOADER (file, n_compacter):
    
    import laspy
    import numpy as np
    
    #file is the path to the .copc.laz file
    # n_compact is a number, meaning we only load in every n datapoint. if n_compact=10 we only load in every 10th point
    
    las = laspy.read(file)

    X_coordinates=np.array([])
    Y_coordinates=np.array([])
    Z_coordinates=np.array([])
    XY_classification=np.array([])
    
    n_points=np.size(las.X)

    Compact_iter=range(0,n_points,n_compacter)


    for i in Compact_iter:
        
        X_coordinates=np.append(X_coordinates,las["X"][i])
        Y_coordinates=np.append(Y_coordinates,las["Y"][i])
        Z_coordinates=np.append(Z_coordinates,las["Z"][i])
        XY_classification=np.append(XY_classification,las["classification"][i])
    
    
    return(X_coordinates,Y_coordinates,Z_coordinates,XY_classification)



def scale(col, min, max):
    range = col.max() - col.min()
    a = (col - col.min()) / range
    return a * (max - min) + min

# Step 1: Create the ternary grid
def create_ternary_grid(n=50):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    valid = (Z >= 0) & (Z <= 1)
    return X[valid], Y[valid], Z[valid]


def predict_mosquitos(water, vegetation, building, points, values):
    if abs(water + vegetation + building - 100) > 1e-6:
        raise ValueError("Percentages must sum to 100")
    
    water, vegetation = water / 100, vegetation / 100
    predicted = griddata(points, values, (water, vegetation), method='cubic')
    return float(predicted)



def Cell_terrain_summery_matrix(X,Y,Z,XY_class,L,n_human,n_mosquito):
    
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import pandas as pd
    
    
    ### --- --------------step 1----------------- --- ####
    ### --- defining the cells, and their borders --- ####
    
    X_min=np.min(X)
    X_max=np.max(X)
    
    X_border_vals=np.linspace(X_min, X_max, num=L+1)
    
    Y_min=np.min(Y)
    Y_max=np.max(Y)
         
    Y_border_vals=np.linspace(Y_min, Y_max, num=L+1)
    
    
    classes=np.unique([1,2,3,4,5,6,9,17,64,65,66])
    n_classes=np.size(np.unique(classes))
    
    ### --- --------------step 2-------------------------------- --- ####
    ### --- defining the the matrices, for the different calsses --- ####
    
    M_Unclassified = np.empty([L,L])
    M_Ground = np.empty([L,L])
    M_Low_Vegitation = np.empty([L,L])
    M_Medium_vegitation = np.empty([L,L])
    M_High_vegitation = np.empty([L,L])
    M_Buildings = np.empty([L,L])
    M_Water= np.empty([L,L])
    M_Bridge_Decks = np.empty([L,L])
    M_64 = np.empty([L,L])
    M_65 = np.empty([L,L])
    M_66 = np.empty([L,L])
    
    
    
    
    
    M_mosquito = np.empty([L,L])

   
    #we define how the theorectical number of points per cell. 
    
    n_points = np.size(X)
    avg_points_per_cell = n_points/(L*L) 
   
    
   
    
    
    ### --- --------------step 3-------------------------------- --- ####
    ### --- looping over every cell, defining its contents------ --- ####
    
    
    for i in range(L):
        temp_x_border=[X_border_vals[i],X_border_vals[i+1]]
        
        for j in range(L):
            temp_y_border=[Y_border_vals[j],Y_border_vals[j+1]]
            
            
            
            
            ##### --- storing data, of all points that exists with the different cells --- ####
            Cell_storrange_vec=np.array([])            
            Cell_Summery_Vec=np.zeros(n_classes)
            
            
            # looping over every single point, storing its data if its within the cell. 
            for XY_iter in range(np.size(X)):
                #checking to see if its within the cell
                if temp_x_border[0] < X[XY_iter] <= temp_x_border[1] and temp_y_border[0] < Y[XY_iter] <= temp_y_border[1]:
                    
    
                    Cell_storrange_vec = np.append(Cell_storrange_vec,XY_class[XY_iter])

            #producing a summery of how many points of each typing there is within the cell

            Cell_Summery_Vec[0] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[0]])
            Cell_Summery_Vec[1] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[1]])
            Cell_Summery_Vec[2] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[2]])
            Cell_Summery_Vec[3] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[3]])
            Cell_Summery_Vec[4] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[4]])
            Cell_Summery_Vec[5] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[5]])
            Cell_Summery_Vec[6] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[6]])
            Cell_Summery_Vec[7] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[7]])
            Cell_Summery_Vec[8] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[8]])
            Cell_Summery_Vec[9] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[9]])
            Cell_Summery_Vec[10] = np.size(Cell_storrange_vec[Cell_storrange_vec==classes[10]])
        
            
            ### --- --------------step 4------------------------------------------------- --- ####
            ### --- store the contents of each cell, in their corresponding matrix ------ --- ####
            
            #scaling with 1 / points in cell. but since not every cell has points in it we make two if branches
            
            #if it has some content, we identify it the following way
            
            if np.sum(Cell_Summery_Vec)>0 : 
            
                ratio_val = 1/ (np.sum(Cell_Summery_Vec))
                
                M_Unclassified[i,j]        = Cell_Summery_Vec[0] * ratio_val
                M_Ground[i,j]              = Cell_Summery_Vec[1] * ratio_val
                M_Low_Vegitation[i,j]      = Cell_Summery_Vec[2] * ratio_val
                M_Medium_vegitation[i,j]   = Cell_Summery_Vec[3] * ratio_val
                M_High_vegitation[i,j]     = Cell_Summery_Vec[4] * ratio_val
                M_Buildings[i,j]           = Cell_Summery_Vec[5] * ratio_val
                M_Water[i,j]               = Cell_Summery_Vec[6] * ratio_val
                M_Bridge_Decks[i,j]        = Cell_Summery_Vec[7] * ratio_val
                M_64[i,j]                  = Cell_Summery_Vec[8] * ratio_val
                M_65[i,j]                  = Cell_Summery_Vec[9] * ratio_val
                M_66[i,j]                  = Cell_Summery_Vec[10]* ratio_val
                
            
            #if the cell is completly empty, we identify it the following way
            #in this instance, we dont know what the should be within the empty cell. 
            #So we simply adress it as unclassified points. It isnt a perfect, but the unclassified section exists for this exact reason.
                
            if np.sum(Cell_Summery_Vec)<=0 :
                
                ratio_val = 1/ (avg_points_per_cell)
                
                M_Unclassified[i,j]        = Cell_Summery_Vec[0]+ (avg_points_per_cell * ratio_val)
                M_Ground[i,j]              = Cell_Summery_Vec[1]
                M_Low_Vegitation[i,j]      = Cell_Summery_Vec[2]
                M_Medium_vegitation[i,j]   = Cell_Summery_Vec[3]
                M_High_vegitation[i,j]     = Cell_Summery_Vec[4]
                M_Buildings[i,j]           = Cell_Summery_Vec[5]
                M_Water[i,j]               = Cell_Summery_Vec[6]
                M_Bridge_Decks[i,j]        = Cell_Summery_Vec[7]
                M_64[i,j]                  = Cell_Summery_Vec[8]
                M_65[i,j]                  = Cell_Summery_Vec[9]
                M_66[i,j]                  = Cell_Summery_Vec[10]
            
            
    ### --- --------------step 5------------------------------------------------- --- ####
    ### --- redifine each matrix, with a gauss filter applied on it. ------ --- ####

 

    
    MM_Unclassified = M_Unclassified
    MM_Ground = M_Ground
    MM_Low_Vegitation = M_Low_Vegitation
    MM_Medium_vegitation = M_Medium_vegitation
    MM_High_vegitation = M_High_vegitation
    MM_Buildings = M_Buildings
    MM_Water = M_Water
    MM_Bridge_Decks =  M_Bridge_Decks
    MM_64 =  M_64
    MM_65 = M_65
    MM_66 = M_66
    
    
    sigma=1
    
    M_Unclassified             = gaussian_filter(M_Unclassified,sigma)
    M_Ground                   = gaussian_filter(M_Ground,sigma)
    M_Low_Vegitation           = gaussian_filter(M_Low_Vegitation,sigma)
    M_Medium_vegitation        = gaussian_filter(M_Medium_vegitation,sigma)
    M_High_vegitation          = gaussian_filter(M_High_vegitation,sigma)
    M_Buildings                = gaussian_filter(M_Buildings,sigma)
    M_Water                    = gaussian_filter(M_Water,sigma)
    M_Bridge_Decks             = gaussian_filter(M_Bridge_Decks,sigma)
    M_64                       = gaussian_filter(M_64,sigma)
    M_65                       = gaussian_filter(M_65,sigma)
    M_65                       = gaussian_filter(M_66,sigma)
    
    
    ### --- --------------step 6------------------------------------------------- --- ####
    ### --- combine all matrices, into one summery matrice. ------ --- ####
    
    
    #here we give postivie scalers (scale>1) for areas that are directly benificial to mosqiitos
    #here we give negative scalers (0<scale<1)
            
    #scalers for  mosquitos
    
    
    #we split it up into 
    
    
    
    
    
    
    Scaler_Mosquito_Unclassified      =0.8    
    Scaler_Mosquito_Ground            =0.8
    Scaler_Mosquito_Low_Vegitation    =1.2
    Scaler_Mosquito_Medium_Vegitation =1.4
    Scaler_Mosquito_High_Vegitation   =1.9
    Scaler_Mosquito_Buildings         =0.8
    Scaler_Mosquito_Water             =1.95
    Scaler_Mosquito_Bridge_Decks      =1.5
    Scaler_Mosquito_64                =0.2
    Scaler_Mosquito_65                =0.2
    Scaler_Mosquito_66                =0.2
    
    
    
    
    
    
    
    
    
    #we produce the summery matrix 
    
    
        
    M_mosquito= (
         
           Scaler_Mosquito_Unclassified * M_Unclassified
         + Scaler_Mosquito_Ground * M_Ground
         + Scaler_Mosquito_Low_Vegitation * M_Low_Vegitation
         + Scaler_Mosquito_Medium_Vegitation * M_Medium_vegitation
         + Scaler_Mosquito_High_Vegitation * M_High_vegitation
         + Scaler_Mosquito_Buildings * M_Buildings
         + Scaler_Mosquito_Water * M_Water
         + Scaler_Mosquito_Bridge_Decks * M_Bridge_Decks
         + Scaler_Mosquito_64 * M_64
         + Scaler_Mosquito_65 * M_65
         + Scaler_Mosquito_66 * M_66
         )
    
    
    
    
    #scalers for  mosquitos
    
    
    Scaler_Human_Unclassified      =1
    Scaler_Human_Ground            =1.2
    Scaler_Human_Low_Vegitation    =0.6
    Scaler_Human_Medium_Vegitation =0.2
    Scaler_Human_High_Vegitation   =0.1
    Scaler_Human_Buildings         =1.95
    Scaler_Human_Water             =0.2
    Scaler_Human_Bridge_Decks      =1.2
    Scaler_Human_64                =0.2
    Scaler_Human_65                =0.2
    Scaler_Human_66                =0.2
    
    
    
    
    #now we normilize rescale the M_human, so it it ranges between 0 and n_humans
    
    
    
    
    M_Human=(
         
           Scaler_Human_Unclassified * M_Unclassified
         + Scaler_Human_Ground * M_Ground
         + Scaler_Human_Low_Vegitation * M_Low_Vegitation
         + Scaler_Human_Medium_Vegitation * M_Medium_vegitation
         + Scaler_Human_High_Vegitation * M_High_vegitation
         + Scaler_Human_Buildings * M_Buildings
         + Scaler_Human_Water * M_Water
         + Scaler_Human_Bridge_Decks * M_Bridge_Decks
         + Scaler_Human_64 * M_64
         + Scaler_Human_65 * M_65
         + Scaler_Human_66 * M_66
         )
    
    
    
    
    
    
    #normalize the dataset
    M_Human = (M_Human - np.min(M_Human)) / (np.max(M_Human) - np.min(M_Human))
    #find the find the sum scale 
    M_human_sum_before=sum([np.sum(M_Human) for i in M_Human])
    M_human_sum_desired = n_human 
    Human_scale_fraction = M_human_sum_desired / M_human_sum_before
    #scaling approapriatly
    M_Human = M_Human * Human_scale_fraction
    
    
    
    #normalize the dataset
    M_mosquito = (M_mosquito - np.min(M_mosquito)) / (np.max(M_mosquito) - np.min(M_mosquito))
    #find the find the sum scale 
    M_mosquito_sum_before=sum([np.sum(M_mosquito) for i in M_mosquito])
    M_mosquito_sum_desired = n_mosquito
    Mosquito_scale_fraction = M_mosquito_sum_desired / M_mosquito_sum_before
    #scaling approapriatly
    M_mosquito = M_mosquito * Mosquito_scale_fraction
    
    
    
   
    

    
    return(M_mosquito,M_Human,MM_Unclassified, MM_Ground,MM_Low_Vegitation,MM_Medium_vegitation,MM_High_vegitation,MM_Buildings,MM_Water,MM_Bridge_Decks,MM_64,MM_65,MM_66)





def Gaussian_Density(X,Y,L, M_mosquito,M_Human,n_mosquitos,n_people):
      
    import numpy as np
    from scipy.stats import multivariate_normal
    
    

    
    
    ### --- --------------step 1--------------------------------------------------------------------- --- ####
    ### --- Scale the values to have the correct relative positions within the gaussian field. ------ --- ####
    
    
    n_nodes=np.size(X)
    
    #noting down the vals 
    X_min=np.min(X)
    X_max=np.max(X)
    
    Y_min=np.min(Y)
    Y_max=np.max(Y)
         
    M_X_min = 0
    M_X_max = L-1
    
    M_Y_min = 0
    M_Y_max = L-1
    

    #scaling the vals.

    OldRange_X = (X_max - X_min)  
    NewRange_X = (M_X_max - M_X_min)  
    X_nodes_scaled = (((X - X_min) * NewRange_X) / OldRange_X) + M_X_min
    
    OldRange_Y = (Y_max - Y_min)  
    NewRange_Y = (M_Y_max - M_Y_min)  
    Y_nodes_scaled = (((Y - Y_min) * NewRange_Y) / OldRange_Y) + M_Y_min
    
    
    
    
    
    
    
    ### --- --------------step 2--------------------------------------------------------------------- --- ####
    ### --- ----------------------------------------------------------------------------------------- --- ####
    
    # loop over each point, and dictate which cell it belongs too.
    # store that cells scalar value, as a seperate list
    
    
    
    #to determain which cell it belong to, we simply round off the nodes
    
    XY_Mosquito_scale_vec=np.empty(n_nodes)
    
    XY_Human_scale_vec=np.empty(n_nodes)
    
    for i in range(n_nodes):
        
        X_rounded=round(X_nodes_scaled[i])
        Y_rounded=round(Y_nodes_scaled[i])
        
        scale_val_Mosquito = M_mosquito[Y_rounded,X_rounded]
        scale_val_human = M_Human[Y_rounded,X_rounded]
        
        XY_Mosquito_scale_vec[i] = (scale_val_Mosquito)
        XY_Human_scale_vec[i] = (scale_val_human)
        
        
        
    ### --- --------------step 3--------------------------------------------------------------------- --- ####
    ### --- ----------------------------------------------------------------------------------------- --- ####
     
    #produce a multivariate_gaussian_landscape for each and every cell. One landscape for Humans, one for Mosquitos.
    
     
    x_vals_mesh = np.arange(0,L)
    y_vals_mesh = np.arange(0,L)
     
    X_mesh,Y_mesh=np.meshgrid(x_vals_mesh,y_vals_mesh)
    
    pos = np.dstack((X_mesh,Y_mesh))
    
    Gaussian_Mosquito_Density_Matrix=np.repeat(0,L)
    Gaussian_Human_Density_Matrix=np.repeat(0,L)
    
    
    
    for i in range(n_nodes):
        
        #set the inputs 
        mu= [X_nodes_scaled[i],Y_nodes_scaled[i]]
        cov = np.array([[1,0],[0,1]])
        #calculate the multivariate_gaussian
        rv = multivariate_normal(mu,cov)
        
        
        Gauss_mesh=rv.pdf(pos)

        
        #produce Gaussian Mosquito Density
        Gaussian_Mosquito_Density_Matrix = np.add((XY_Mosquito_scale_vec[i])*Gauss_mesh,Gaussian_Mosquito_Density_Matrix)
    
        
        #produce Gaussian Human Density
        Gaussian_Human_Density_Matrix = np.add((XY_Human_scale_vec[i])*Gauss_mesh,Gaussian_Human_Density_Matrix)
 
    
    
    
    
    Gaussian_Human_density_M = Gaussian_Human_Density_Matrix
    Gaussian_Mosquito_density_M = Gaussian_Mosquito_Density_Matrix
    
    return(Gaussian_Mosquito_density_M,Gaussian_Human_density_M)






def AdjacencyAndTransmission_matrix (X,Y,L,Critical_Distance):
    
    import numpy as np
    from scipy.spatial.distance import euclidean
    import math
    
    n_nodes=np.size(X)
    
    #noting down the vals 
    X_min=np.min(X)
    X_max=np.max(X)
    
    Y_min=np.min(Y)
    Y_max=np.max(Y)
         
    M_X_min = 0
    M_X_max = L-1
    
    M_Y_min = 0
    M_Y_max = L-1
    

    #scaling the vals.

    OldRange_X = (X_max - X_min)  
    NewRange_X = (M_X_max - M_X_min)  
    X_nodes_scaled = (((X - X_min) * NewRange_X) / OldRange_X) + M_X_min
    
    OldRange_Y = (Y_max - Y_min)  
    NewRange_Y = (M_Y_max - M_Y_min)  
    Y_nodes_scaled = (((Y - Y_min) * NewRange_Y) / OldRange_Y) + M_Y_min
    # here it is important to know the size and shape of our data
    # the LAZ data is a 1km by 1km square. and X_max-X_min is ~100 000
    # meaning our data is in cm, not meters. 
    # our input critical distance is in meters, and scaling must be done accordingly (*100)
    
    Critical_range_X_old=X_max-X_min
    Critical_range_X_new=L-1
    
    Distance_Scale= Critical_range_X_new / Critical_range_X_old
    Critical_Distance_Scaled = (100*Critical_Distance) * Distance_Scale
    
    
    #creating a proximity matrix
    
    Proximity_Matrix = np.zeros(shape=[n_nodes,n_nodes])
    Transmission_Matrix = np.zeros(shape=[n_nodes,n_nodes])

    
    for i in range(n_nodes):
        point_i=[X_nodes_scaled[i],Y_nodes_scaled[i]]
        
        
        for j in range(n_nodes):
            point_j=[X_nodes_scaled[j],Y_nodes_scaled[j]]
        
            
            #calculating the euclidian distance
            distance_ij=euclidean(point_i,point_j)
            
            
            if distance_ij < Critical_Distance_Scaled and i!=j:
                
                Proximity_Matrix[i,j]=1
                #now we loop over every cell. and see if exists within the "circle overlap" between i and j.
                #we simply calculate the area of the ciricle overlap, produced by the two circles from the points and critical radai. 
                
                overlap_Val=(2 * Critical_Distance_Scaled**2) * (math.acos(distance_ij / (2 * Critical_Distance_Scaled)) - (distance_ij / (2 * Critical_Distance_Scaled)) * math.sqrt(1 - (distance_ij / (2 * Critical_Distance_Scaled))**2))
                
                #we now need to incorporate a way to increase transmission if the conditions are good. 
                #for this, we will use the difference in peolpe in point A and B
                #if there are more people in point i than in point j, the mosquitos in point i should has an increased insentive to fly to point j
            
                Transmission_Matrix[i,j] = overlap_Val 
            if i==j:
                
                Proximity_Matrix[i,j]=1
                overlap_Val = math.pi * Critical_Distance_Scaled**2
                Transmission_Matrix[i,j] = overlap_Val 

    print(Critical_Distance_Scaled)
    return(Proximity_Matrix, Transmission_Matrix)



def Mosquito_Human_Count(X,Y,L, Mosquito_M, Human_M):
    
    n_nodes=np.size(X)
    
    #noting down the vals 
    X_min=np.min(X)
    X_max=np.max(X)
    
    Y_min=np.min(Y)
    Y_max=np.max(Y)
         
    M_X_min = 0
    M_X_max = L-1
    
    M_Y_min = 0
    M_Y_max = L-1
    

    #scaling the vals.

    OldRange_X = (X_max - X_min)  
    NewRange_X = (M_X_max - M_X_min)  
    X_nodes_scaled = (((X - X_min) * NewRange_X) / OldRange_X) + M_X_min
    
    OldRange_Y = (Y_max - Y_min)  
    NewRange_Y = (M_Y_max - M_Y_min)  
    Y_nodes_scaled = (((Y - Y_min) * NewRange_Y) / OldRange_Y) + M_Y_min
    
    X_nodes_scaled_rounded=np.round(X_nodes_scaled)
    Y_nodes_scaled_rounded=np.round(Y_nodes_scaled)
    
    Mosquito_count=np.zeros(n_nodes)
    Human_count= np.zeros(n_nodes)
    
    
    
    
    for i in range(n_nodes):
        
        xi = int(X_nodes_scaled_rounded[i])
        yi = int(Y_nodes_scaled_rounded[i])
        Mosquito_count[i] = Mosquito_M[xi,yi]
        Human_count[i]    =    Human_M[xi,yi]
     
    
    
    
    #now we normilize both, so they sum to 1
    
    from sklearn import preprocessing
    
    
    normalized_Mosquito_count = preprocessing.normalize([Mosquito_count])
    normalized_Human_count = preprocessing.normalize([Human_count])
    
    
    return(normalized_Mosquito_count,normalized_Human_count)
       
       
    
    

    
    
    


def PlotStuff(X,Y,L,M_cell_summery,Adjacency_matrix):
    
    import matplotlib.pyplot as plt
    
    
    #noting down the vals 
    X_min=np.min(X)
    X_max=np.max(X)
    
    Y_min=np.min(Y)
    Y_max=np.max(Y)
         
    M_X_min = 0
    M_X_max = L-1
    
    M_Y_min = 0
    M_Y_max = L-1
    

    #scaling the vals.

    OldRange_X = (X_max - X_min)  
    NewRange_X = (M_X_max - M_X_min)  
    X_nodes_scaled = (((X - X_min) * NewRange_X) / OldRange_X) + M_X_min
    
    OldRange_Y = (Y_max - Y_min)  
    NewRange_Y = (M_Y_max - M_Y_min)  
    Y_nodes_scaled = (((Y - Y_min) * NewRange_Y) / OldRange_Y) + M_Y_min

    
    
    fig,ax=plt.subplots(1,3)
    
    ax[0].set(xlim=(0,L-1),ylim=(L-1,0))
    ax[0].matshow(M_cell_summery)
    ax[0].scatter(X,Y, c="red", s=0.8)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)


    ax[1].set(xlim=(0,L-1),ylim=(L-1,0))
    ax[1].set_aspect('equal')
    
    for i in range(np.size(X_nodes_scaled)): #X
        for j in range(np.size(Y_nodes_scaled)): #Y
            
            if Adjacency_matrix[i,j]==1 and i!=j:
                
                ax[1].plot([X_nodes_scaled[i],X_nodes_scaled[j]],[Y_nodes_scaled[i], Y_nodes_scaled[j]],c="blue",linewidth=0.2)
    
    ax[1].scatter(X_nodes_scaled,Y_nodes_scaled, c="red", s=1.2)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)


    plt.show()
    
    
    

    return("returning plot")
    


    
    
    









    




