# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:15:58 2024

@author: Kristian Juul Rasmussen
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import plotly.figure_factory as ff

# Example dataset
data = pd.DataFrame({
  #Site A, Site B, Site C, Site D, Site E, Site F, Site G, Site H, Site H, Site I, Site J, site G, corners
  
    'vegetation':   [68.97, 66.20, 60.20, 88.64, 96.35 ,81.71,  47.875, 24.77, 55.10, 17.38, 100., 0.  , 0.], #pecentage vegitation
    'water':        [7.58 , 9.02 , 2.93 , 4.25 , 0.00  ,4.78 ,  36.336, 62.14, 31.92, 76.22, 0.  , 100., 0.],        # Percentage of water
    'buildings':    [23.45, 24.78, 36.87, 7.11 , 3.65  ,13.51,  15.789, 13.09, 12.98, 6.40 , 0.  , 0.   , 100.],    # Percentage of buildings
    'mosquitos':    [56,       44,   106,   22 , 10    ,69   ,  377  ,   177, 35   , 232   , 1.  , 1.  , 1.]  # Mosquito count
    
})



######## alternative method????


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

# Example usage of the function
print(predict_mosquitos(33, 33, 34))


#water, vegetation, building
print(predict_mosquitos(1, 1, 98))



#plotting the ternary plot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# Assuming you've already created X, Y, and grid_mosquitos as in the previous code

def to_equilateral(x, y):
    return x + y/2, y*np.sqrt(3)/2

X_eq, Y_eq = to_equilateral(X, Y)

fig, ax = plt.subplots(figsize=(12, 10))

tri = mtri.Triangulation(X_eq, Y_eq)
contour = ax.tricontourf(tri, grid_mosquitos, levels=20, cmap='viridis')

cbar = plt.colorbar(contour,orientation = 'horizontal')
cbar.set_label('Predicted Mosquito Count')

data_X_eq, data_Y_eq = to_equilateral(data['water']/100, data['vegetation']/100)
scatter = ax.scatter(data_X_eq, data_Y_eq, 
                     c=data['mosquitos'], s=50, edgecolor='k', cmap='viridis')

ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Ternary Plot of Mosquito Counts (Equilateral)')

# Define colors for each component
water_color = 'blue'
vegetation_color = 'green'
building_color = 'red'

# Add colored labels near the corners
ax.text(1.05, -0.05, 'Water', ha='left', va='top', color="blue", fontsize=14, fontweight='bold')
ax.text(-0.05, -0.05, 'buildings', ha='right', va='top', color="red", fontsize=14, fontweight='bold')
ax.text(0.5, np.sqrt(3)/2 + 0.05, 'vegetation', ha='center', va='bottom', color="green", fontsize=14, fontweight='bold')

# Add colored arrows for each axis
arrow_props = dict(arrowstyle='->', lw=3, mutation_scale=20)

# Water arrow (bottom edge, left to right)
ax.annotate('', xy=(0, 0), xytext=(1, 0), 
            arrowprops=dict(arrowstyle='->', color="red", lw=3, mutation_scale=20))

# Vegetation arrow (left edge, bottom to top)
ax.annotate('', xy=(0.5, np.sqrt(3)/2), xytext=(0, 0), 
            arrowprops=dict(arrowstyle='->', color="green", lw=3, mutation_scale=20))

# Building arrow (right edge, top to bottom)
ax.annotate('', xy=(1, 0), xytext=(0.5, np.sqrt(3)/2), 
            arrowprops=dict(arrowstyle='->', color="blue", lw=3, mutation_scale=20))


# Add triangle outline
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2)

plt.tight_layout()
plt.show()




import plotly.offline as pyo
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

water=data["water"]
buildings=data["buildings"]
vegetation=data["vegetation"]
Mosquitos=data["mosquitos"]


fig = ff.create_ternary_contour(np.array([vegetation, buildings,water]), Mosquitos,
                                pole_labels=['vegetation', 'buildings', 'water'],
                                interp_mode='cartesian',
                                ncontours=10,
                                colorscale='Viridis',
                                showscale=True,
                                showmarkers=True)



fig.show()



pyo.plot(fig)
