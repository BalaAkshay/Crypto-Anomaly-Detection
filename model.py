import pandas as pd
from sklearn.ensemble import IsolationForest



def detect_anomalies_isolation_forest(data, features, contamination):
    
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(data[features])  
    print("Model fitting complete.")
    
    data["Anomaly"] = model.predict(data[features])  
    anomalies_indices = data[data["Anomaly"] == -1].index  
    print(f"Detected {len(anomalies_indices)} anomalies.")
    
    return anomalies_indices
    