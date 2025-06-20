import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# === Custom Modules ===
from data_collection import fetch_binance_data
from preprocessing import load_data, calculate_features
from feature_engineering import (
    apply_minmax_scaling,
    apply_standard_scaling,
    apply_winsorization,
    apply_log_transformation
)
from visualization import (
    plot_anomalies_if,
    plot_anomalies_auto,
    plot_reconstruction_error,
    visualize_data,
    plot_yearly_anomalies_auto,
    plot_yearly_anomalies_if
)
from isolation_forest import detect_anomalies_isolation_forest
from autoencoder import (
    build_autoencoder,
    train_autoencoder,
    detect_anomalies_autoencoder
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# === Input function for selecting crypto ===
def get_crypto_symbol():
    print("Select a cryptocurrency to analyze:")
    print("1. Bitcoin (BTC)")
    print("2. Dogecoin (DOGE)")
    print("3. Ethereum (ETH)")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == '1':
        return "BTCUSDT"
    elif choice == '2':
        return "DOGEUSDT"
    elif choice == '3':
        return "ETHUSDT"
    else:
        print("Invalid choice. Defaulting to Dogecoin (DOGEUSDT).")
        return "DOGEUSDT"

def main():
    # === Step 0: Choose crypto ===
    symbol = get_crypto_symbol()
    interval = "1d"
    start_date = "1 Jan 2018"
    end_date = None

    # === Step 1: Fetch Data ===
    data = fetch_binance_data(symbol, interval, start_date, end_date)
    data.to_csv(f"{symbol}_historical_data.csv", index=False)
    print("Data fetched and saved successfully!")

    # === Step 2: Load & Feature Engineering ===
    data = load_data(f"{symbol}_historical_data.csv")
    data = calculate_features(data)
    data_scaled = data.copy()

    # Feature Scaling
    bounded_columns = ["RSI", "Volume Change"]
    data_scaled = apply_minmax_scaling(data_scaled, bounded_columns)

    unbounded_columns = ["Close", "Daily Return", "Volatility_7", "Volatility_30"]
    data_scaled = apply_standard_scaling(data_scaled, unbounded_columns)
    data_scaled = apply_winsorization(data_scaled, bounded_columns + unbounded_columns)

    skewed_columns = [
        "Quote Asset Volume",
        "Number of Trades",
        "Taker Buy Base Asset volume",
        "Taker Buy Quote asset Volume"
    ]
    data_scaled = apply_log_transformation(data_scaled, skewed_columns)
    data_scaled.to_csv(f"{symbol}_processed_data.csv", index=False)
    print("Data preprocessing and feature engineering completed successfully!")

    # === Step 3: Isolation Forest ===
    features_if = ["Close", "Volume", "RSI", "SMA_30", "Daily Return"]
    data_subset_if = data_scaled[features_if].dropna()

    print("Detecting anomalies using Isolation Forest...")
    anomalies_indices_if, pump_indices_if, dump_indices_if = detect_anomalies_isolation_forest(
        data_subset_if, features_if, contamination=0.03
    )

    anomalies_if = data.loc[anomalies_indices_if].copy()
    anomalies_if["Anomaly_Type"] = ["Pump" if idx in pump_indices_if else "Dump" for idx in anomalies_indices_if]
    anomalies_if.to_csv(f"anomalies_if_{symbol}.csv", index=False)
    print("Anomaly detection using Isolation Forest completed and results saved!")

    # === Step 4: Autoencoder ===
    features_auto = ['Close', 'Daily Return', 'Volume', 'RSI', 'MACD', 'SMA_7']
    data_subset_auto = data_scaled[features_auto].dropna()

    X_train, X_test = train_test_split(data_subset_auto, test_size=0.18, random_state=2)
    input_dim = X_train.shape[1]

    autoencoder = build_autoencoder(input_dim, encoding_dim=32)
    train_autoencoder(autoencoder, X_train, epochs=70, batch_size=32, validation_data=(X_test, X_test))

    print("Detecting anomalies using Autoencoder....")
    anomalies_auto, reconstruction_error, threshold, labels = detect_anomalies_autoencoder(
        autoencoder, X_test.values, threshold_percentile=95, return_threshold=True
    )

    anomalies_indices_auto = np.where(anomalies_auto)[0]
    anomalies_data = data.iloc[-len(X_test):].iloc[anomalies_indices_auto].copy()
    anomalies_data['Anomaly_Type'] = labels[anomalies_auto]
    anomalies_data.to_csv(f"anomalies_autoencoder_{symbol}.csv", index=False)
    print("Anomaly detection using Autoencoder completed and results saved!")

    # === Step 5: Visualization ===
    print("Visualizing results...")

    visualize_data(data, symbol)

    # Isolation Forest visualizations
    plot_anomalies_if(data, pump_indices_if, symbol)
    plot_anomalies_if(data, dump_indices_if, symbol)

    # Autoencoder visualizations
    plot_anomalies_auto(data, X_test, anomalies_auto, labels, symbol)

    # Yearly Plots
    for year in range(2020, 2026):
        plot_yearly_anomalies_auto(data, X_test, anomalies_auto, labels, year, symbol)
        plot_yearly_anomalies_if(data, pump_indices_if, year, "Pump", "green", "x", symbol)
        plot_yearly_anomalies_if(data, dump_indices_if, year, "Dump", "red", "o", symbol)

    # Reconstruction Error
    plot_reconstruction_error(reconstruction_error, threshold)

    print("All visualizations complete!")

if __name__ == "__main__":
    main()
