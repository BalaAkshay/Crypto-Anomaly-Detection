import pandas as pd
from sklearn.ensemble import IsolationForest



def detect_anomalies_isolation_forest(data, features, contamination=0.01):
    
    model = IsolationForest(contamination=contamination, random_state=42)
    data["Anomaly"] = model.fit_predict(data[features])
    
    anomalies = data[data["Anomaly"] == -1]
    
    return anomalies