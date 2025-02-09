import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def apply_minmax_scaling(data, columns):
   
    scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def apply_standard_scaling(data, columns):
    
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

def apply_winsorization(data, columns):
    
    for column in columns:
        data[column] = winsorize(data[column], limits=[0.05, 0.05])
    return data


def apply_log_transformation(data, columns):
    
    data[columns] = np.log1p(data[columns])
    return data
    