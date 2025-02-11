from data_collection import fetch_binance_data
from preprocessing import load_data, calculate_features
from feature_engineering import apply_minmax_scaling, apply_standard_scaling, apply_winsorization, apply_log_transformation
from visualization import visualize_data
from model import detect_anomalies_isolation_forest

def main():
    # Step 1: Fetch data from Binance API
    symbol = "BTCUSDT"
    interval = "1d"
    start_date = "1 Jan 2018"
    end_date = "1 Jan 2025"
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

    visualize_data(data)

    print("Detecting anomalies using Isolation Forest...")
    anomalies_if = detect_anomalies_isolation_forest(data_scaled, ['Close', 'Volume'])
    print(f"Isolation Forest Anomalies: {len(anomalies_if)}")

    anomalies_if.to_csv("anomalies_isolation_forest.csv", index=False)

if __name__ == "__main__":
    main()