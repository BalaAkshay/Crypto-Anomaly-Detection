import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from data_collection import fetch_binance_data
from preprocessing import load_data, calculate_features
from feature_engineering import apply_minmax_scaling, apply_standard_scaling, apply_winsorization, apply_log_transformation
from visualization import plot_anomalies_if, plot_anomalies_auto
from isolation_forest import detect_anomalies_isolation_forest
from autoecoder import build_autoencoder, train_autoencoder, detect_anomalies_autoencoder

np.random.seed(42)
tf.random.set_seed(42)


def main():
    # Step 1: Fetch data from Binance API
    symbol = "BTCUSDT"
    interval = "1d"
    start_date = "1 Jan 2018"
    end_date = None
    data = fetch_binance_data(symbol, interval, start_date, end_date)
    data.to_csv("BTCUSDT_historical_data.csv", index=False)
    print("Data fetched and saved successfully!")

    # Step 2: Load and preprocess data
    data = load_data("BTCUSDT_historical_data.csv")
    data = calculate_features(data)
    data_scaled = data.copy()

    # Step 3: Feature engineering
    bounded_columns = ["RSI", "Volume Change"]
    data_scaled = apply_minmax_scaling(data_scaled, bounded_columns)
   
    unbounded_columns = ["Close", "Daily Return", "Volatility_7", "Volatility_30"]
    data_scaled = apply_standard_scaling(data_scaled, unbounded_columns)

    data_scaled = apply_winsorization(data_scaled, bounded_columns + unbounded_columns)

    skewed_columns = ["Quote Asset Volume", "Number of Trades", "Taker Buy Base Asset volume", "Taker Buy Quote asset Volume"]
    data_scaled = apply_log_transformation(data_scaled, skewed_columns)

    
    # Save the final processed data
    data_scaled.to_csv("BTCUSDT_processed_data.csv", index=False)
    print("Data preprocessing and feature engineering completed successfully!")

    # Step 4.1: Isolation Forest Model
    advanced_features_if = ["Bollinger_Upper", "Bollinger_Lower", "Signal_Line"]
    basic_features_if = ["Close", "Volume", "RSI", "SMA_30", "Daily Return"]
    features_if =   basic_features_if

    data_subset_if = data_scaled[features_if].dropna()

    print("Detecting anomalies using Isolation Forest...")
    anomalies_indices_if = detect_anomalies_isolation_forest(data_subset_if, features_if, 0.03)

    # Save results
    anomalies_if = data.loc[anomalies_indices_if]  # Get anomaly rows from original data
    anomalies_if.to_csv("anomalies_isolation_forest.csv", index=False)

    print("!!Anomaly detection using Isolation Forest completed and results saved!!")



    # Step 4.2: Autoencoder Model
    features_auto = ['Close', 'Daily Return', 'Volume', 'RSI', 'MACD', 'SMA_7']

    data_subset_auto = data_scaled[features_auto].dropna()

    X_train, X_test = train_test_split(data_subset_auto, test_size = 0.18, random_state = 2)

    input_dim = X_train.shape[1]

    autoencoder = build_autoencoder(input_dim, encoding_dim = 32)

    history = train_autoencoder(autoencoder, X_train, epochs = 70, batch_size = 32, validation_data = (X_test, X_test))

    print("Detecting anomalies using Autoencoder....")
    anomalies_auto, reconstruction_error = detect_anomalies_autoencoder(autoencoder, X_test, threshold_percentile = 95)

    # Save results
    anomalies_indices_auto = np.where(anomalies_auto)[0]
    anomalies_data = data.iloc[-len(X_test):].iloc[anomalies_indices_auto]
    anomalies_data.to_csv("anomalies_autoencoder.csv", index=False)
    
    print("!!Anomaly detection using Autoencoder completed and results saved!!")

    # Visualiation
    #plot_anomalies_if(data, anomalies_indices_if, title="Anomalies in BTCUSDT Closing Price(Isolation Forest)")
    plot_anomalies_auto(data, X_test, anomalies_indices_auto, title="Anomalies in Closing Price(Autoencoder)")
    


if __name__ == "__main__":
    main()