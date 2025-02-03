from data_collection import fetch_binance_data
from preprocessing import load_data, calculate_features
from feature_engineering import apply_minmax_scaling, apply_standard_scaling, apply_winsorization, apply_log_transformation

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

    # Step 3: Feature engineering
    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "Daily Return", "SMA_7", "SMA_30", "Volatility_7", "Volatility_30"]
    data = apply_minmax_scaling(data, numeric_columns)
    data = apply_standard_scaling(data, numeric_columns)
    data = apply_winsorization(data, numeric_columns)
    data = apply_log_transformation(data, numeric_columns)

    # Save the final processed data
    data.to_csv("BTCUSDT_processed_data.csv", index=False)
    print("Data processing and feature engineering completed successfully!")

if __name__ == "__main__":
    main()