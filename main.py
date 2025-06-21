import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from data_collection import fetch_binance_data
from preprocessing import load_data, calculate_features
from feature_engineering import apply_minmax_scaling, apply_standard_scaling, apply_winsorization, apply_log_transformation
from visualization import plot_anomalies_if, plot_anomalies_auto, plot_anomalies_attention
from isolation_forest import detect_anomalies_isolation_forest
from autoencoder import build_autoencoder, train_autoencoder, detect_anomalies_autoencoder
from attention import build_attention_autoencoder, train_attention_ae, detect_anomalies_attention_ae, SelfAttention
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

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
    bounded_columns = ["RSI", "Volume Change", "Bollinger_Upper", "Bollinger_Lower", "Signal_Line"]
    data_scaled = apply_minmax_scaling(data_scaled, bounded_columns)
   
    unbounded_columns = ["Close", "Daily Return", "Volatility_7", "Volatility_30", "SMA_30", "SMA_7", "MACD", "EMA_12", "EMA_26"]
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
    anomalies_indices_if = detect_anomalies_isolation_forest(data_subset_if, features_if, 0.05)

    # Save results
    anomalies_if = data.loc[anomalies_indices_if]  # Get anomaly rows from original data
    anomalies_if.to_csv("anomalies_isolation_forest.csv", index=False)

    print("!!Anomaly detection using Isolation Forest completed and results saved!!")



    # Step 4.2: Autoencoder Model

    features_auto = ['Close', 'Daily Return', 'Volume', 'RSI', 'MACD', 'SMA_7', "Bollinger_Upper", "Bollinger_Lower", "Signal_Line", "EMA_12", "EMA_26"]

    data_subset_auto = data_scaled[features_auto].dropna()

    X_train, X_test = train_test_split(data_subset_auto, test_size = 0.1, random_state = 42)

    input_dim = X_train.shape[1]
    print("input_dim: ", input_dim)

    autoencoder = build_autoencoder(input_dim, encoding_dim = 8)

    #print(autoencoder.summary())
    
    history = train_autoencoder(autoencoder, X_train, epochs = 10, batch_size = 64, validation_data = (X_test, X_test))

    print("Detecting anomalies using Autoencoder....")
    anomalies_auto, reconstruction_error = detect_anomalies_autoencoder(autoencoder, X_test, threshold_percentile = 95)

    # Save results
    anomalies_indices_auto = np.where(anomalies_auto)[0]
    anomalies_data = data.iloc[-len(X_test):].iloc[anomalies_indices_auto]
    anomalies_data.to_csv("anomalies_autoencoder.csv", index=False)
    
    print("!!Anomaly detection using Autoencoder completed and results saved!!")

    # Step 4.3: Attention_Autoencoder Model

    features_auto = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_7',  "Signal_Line", "EMA_12"]

    data_subset_auto = data_scaled[features_auto].dropna()

    X_train, X_test = train_test_split(data_subset_auto, test_size = 0.1, random_state = 42)

    input_dim = X_train.shape[1]
    print("input_dim: ", input_dim)

    attention_ae = build_attention_autoencoder(input_dim, encoding_dim = 6, time_steps = 7)

    history_att = train_attention_ae(attention_ae, X_train, epochs=300, batch_size=32, validation_data=(X_test, X_test))
    anomalies_attention, reconstruction_error_attention = detect_anomalies_attention_ae(attention_ae, X_test, threshold_percentile=95)

    # Save results
    anomalies_indices_attention = np.where(anomalies_attention)[0]
    anomalies_data_attention = data.iloc[-len(X_test):].iloc[anomalies_indices_attention]
    anomalies_data_attention.to_csv("anomalies_attention.csv", index=False)

    print("!!Anomaly detection using Attention Autoencoder completed and results saved!!")

    print(autoencoder.summary())
    print(attention_ae.summary())
    # Visualiation
    plot_anomalies_if(data, anomalies_indices_if, title="Anomalies in BTCUSDT Closing Price(Isolation Forest)")
    plot_anomalies_auto(data, X_test, anomalies_indices_auto, title="Anomalies in Closing Price(Autoencoder)")
    plot_anomalies_attention(data, X_test, anomalies_indices_attention, title="Anomalies in Closing Price(Attention_ae)")



if __name__ == "__main__":
    main()