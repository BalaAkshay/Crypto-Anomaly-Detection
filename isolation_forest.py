from sklearn.ensemble import IsolationForest


def detect_anomalies_isolation_forest(data, features, contamination):
    """
    Detect anomalies using Isolation Forest and classify them into pump and dump.

    Args:
        data (pd.DataFrame): DataFrame containing input features and 'Daily Return'.
        features (list): List of column names to use for Isolation Forest.
        contamination (float): Proportion of outliers in the data set.

    Returns:
        anomalies_indices (pd.Index): Indices of all detected anomalies.
        pump_indices (list): Indices where anomaly and Daily Return > 0.
        dump_indices (list): Indices where anomaly and Daily Return <= 0.
    """
    # Fit model
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(data[features])
    print("Isolation Forest model fitting complete.")

    # Detect anomalies (-1 indicates anomaly)
    data['Anomaly'] = model.predict(data[features])
    anomalies_indices = data[data['Anomaly'] == -1].index
    print(f"Detected {len(anomalies_indices)} anomalies.")

    # Classify anomalies into pump vs dump by 'Daily Return'
    pump_indices = [idx for idx in anomalies_indices if data.loc[idx, 'Daily Return'] > 0]
    dump_indices = [idx for idx in anomalies_indices if data.loc[idx, 'Daily Return'] <= 0]

    print(f"Pump anomalies: {len(pump_indices)}, Dump anomalies: {len(dump_indices)}")

    return anomalies_indices, pump_indices, dump_indices
