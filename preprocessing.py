import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    data = pd.read_csv(filepath)
    data["Open Time"] = pd.to_datetime(data["Open Time"])
    data.set_index("Open Time", inplace=True)
    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    return data


def calculate_features(data):
    
    # Daily Return
    data["Daily Return"] = data["Close"].pct_change()

    # Moving Averages
    data["SMA_7"] = data["Close"].rolling(window=7).mean()
    data["SMA_30"] = data["Close"].rolling(window=30).mean()

    # Volatility
    data["Volatility_7"] = data["Daily Return"].rolling(window=7).std()
    data["Volatility_30"] = data["Daily Return"].rolling(window=30).std()

    # Volume Change
    data["Volume Change"] = data["Volume"].pct_change()

    # Bollinger Bands
    data["Bollinger_Upper"] = data["SMA_30"] + (2 * data["Close"].rolling(window=30).std())
    data["Bollinger_Lower"] = data["SMA_30"] - (2 * data["Close"].rolling(window=30).std())

    # RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Log Returns
    data["Log Return"] = np.log(data["Close"] / data["Close"].shift(1))

    # Handle Outliers
    data["Z_Score"] = (data["Close"] - data["Close"].mean()) / data["Close"].std()
    data = data[(data["Z_Score"].abs() <= 3)]

    data.dropna(inplace=True)
    return data