import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def apply_minmax_scaling(data, columns):
    
    #Apply Min-Max scaling to specified columns.

    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def apply_standard_scaling(data, columns):
    
    #Apply Standard Scaling (Z-score normalization) to specified columns.
    
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def apply_winsorization(data, columns, limits=(0.05, 0.05)):
    
    #Apply Winsorization to specified columns to handle outliers.
    
    for col in columns:
        data[col] = winsorize(data[col], limits=limits)
    return data

def apply_log_transformation(data, columns):
    
    #Apply log transformation to specified columns.
    
    for col in columns:
        data[col] = np.log1p(data[col])  # Use log1p to handle zero values
    return data