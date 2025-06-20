import matplotlib.pyplot as plt

# === 1. Visualize Price and Moving Averages ===
def visualize_data(data, symbol="CRYPTO", show_sma=True):
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")

    if show_sma:
        if "SMA_7" in data.columns:
            plt.plot(data.index, data["SMA_7"], label="7-Day SMA", color="orange", linestyle='--')
        if "SMA_30" in data.columns:
            plt.plot(data.index, data["SMA_30"], label="30-Day SMA", color="green", linestyle='--')

    plt.title(f"{symbol} Close Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === 2. Isolation Forest Anomaly Plot ===
def plot_anomalies_if(data, anomalies_indices, symbol="CRYPTO", title=None):
    if title is None:
        title = f"{symbol} Anomalies in Closing Price (Isolation Forest)"
        
    plt.figure(figsize=(14, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")

    anomalies = data.loc[anomalies_indices]
    plt.scatter(anomalies.index, anomalies["Close"], color="red", label="Anomalies", marker="x")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === 3. Autoencoder Anomaly Plot with Pump/Dump Distinction ===
def plot_anomalies_auto(data, X_test, anomalies_mask, labels, symbol="CRYPTO", title=None):
    if title is None:
        title = f"{symbol} Anomalies in Closing Price (Autoencoder)"

    plt.figure(figsize=(14, 6))

    test_data = data.iloc[-len(X_test):].copy()
    plt.plot(test_data.index, test_data["Close"], label="Close Price", color="blue")

    pump_indices = test_data.index[(anomalies_mask) & (labels == 'Pump')]
    dump_indices = test_data.index[(anomalies_mask) & (labels == 'Dump')]

    plt.scatter(pump_indices, test_data.loc[pump_indices, "Close"], color="green", label="Pump (↑)", marker="x")
    plt.scatter(dump_indices, test_data.loc[dump_indices, "Close"], color="red", label="Dump (↓)", marker="o")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === 4. Reconstruction Error Plot ===
def plot_reconstruction_error(errors, threshold, title="Reconstruction Error with Threshold"):
    plt.figure(figsize=(12, 4))
    plt.plot(errors, label="Reconstruction Error", color="gray")
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === 5. Yearly Anomaly Plots for Autoencoder ===
def plot_yearly_anomalies_auto(data, X_test, anomalies_mask, labels, year, symbol="CRYPTO"):
    test_data = data.iloc[-len(X_test):].copy()
    yearly_data = test_data[test_data.index.year == year]
    if yearly_data.empty:
        return

    plt.figure(figsize=(14, 6))
    plt.plot(yearly_data.index, yearly_data["Close"], label="Close Price", color="blue")

    anomaly_mask = anomalies_mask[-len(X_test):]
    pump_indices = yearly_data.index[(anomaly_mask & (labels == 'Pump')) & (yearly_data.index.isin(yearly_data.index))]
    dump_indices = yearly_data.index[(anomaly_mask & (labels == 'Dump')) & (yearly_data.index.isin(yearly_data.index))]

    plt.scatter(pump_indices, yearly_data.loc[pump_indices, "Close"], color="green", label="Pump (↑)", marker="x")
    plt.scatter(dump_indices, yearly_data.loc[dump_indices, "Close"], color="red", label="Dump (↓)", marker="o")

    plt.title(f"Autoencoder Anomalies in {symbol} ({year})")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# === 6. Yearly Anomaly Plots for Isolation Forest ===
def plot_yearly_anomalies_if(data, anomaly_indices, year, label, color, marker, symbol="CRYPTO"):
    yearly_data = data[data.index.year == year]
    anomaly_year_indices = [idx for idx in anomaly_indices if idx in yearly_data.index]
    if not anomaly_year_indices:
        return

    plt.figure(figsize=(14, 6))
    plt.plot(yearly_data.index, yearly_data["Close"], label="Close Price", color="blue")
    plt.scatter(anomaly_year_indices, data.loc[anomaly_year_indices, "Close"], color=color, label=label, marker=marker)
    plt.title(f"Isolation Forest {label}s in {symbol} ({year})")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

