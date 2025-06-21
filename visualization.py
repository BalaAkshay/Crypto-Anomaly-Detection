import matplotlib.pyplot as plt

def visualize_data(data):
   
   # Visualizes close price and moving averages

    plt.figure(figsize=(12, 6))
    plt.plot(data["Close"], label="Close Price")
    plt.plot(data["SMA_7"], label="7-Day SMA")
    plt.plot(data["SMA_30"], label="30-Day SMA")        
    plt.legend()
    plt.title("Close Price and Moving Averages")
    plt.show()



def plot_anomalies_if(data, anomalies_indices, title="Anomalies in Closing Price(Isolation Forest)"):
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    #plt.plot(data.index, data["SMA_7"], label="7-Day SMA", color ='green')
    #plt.plot(data.index, data["SMA_30"], label="30-Day SMA", color ='yellow')

    anomalies = data.loc[anomalies_indices]
    plt.scatter(anomalies.index, anomalies["Close"], color="red", label="Anomalies", marker="x")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()


def plot_anomalies_auto(data, X_test, anomalies_indices, title="Anomalies in Closing Price(Autoencoder)"):
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    plt.scatter(
        data.index[-len(X_test):][anomalies_indices],
        data.iloc[-len(X_test):]["Close"].iloc[anomalies_indices],
        color="red", label="Anomalies", marker="x"
    )
    plt.title("Anomalies in BTCUSDT Closing Price (Autoencoder)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()


def plot_anomalies_attention(data, X_test, anomalies_indices, title="Anomalies in Closing Price(Attention_ae)"):
    
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    plt.scatter(
        data.index[-len(X_test):][anomalies_indices],
        data.iloc[-len(X_test):]["Close"].iloc[anomalies_indices],
        color="red", label="Anomalies", marker="x"
    )
    plt.title("Anomalies in BTCUSDT Closing Price (Attention_ae)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()