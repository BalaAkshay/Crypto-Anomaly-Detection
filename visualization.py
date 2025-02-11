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



def plot_anomalies(data, anomalies_indices, title="Anomalies in Closing Price"):
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(data.index, data["Close"], label="Close Price", color="blue")
    plt.plot(data.index, data["SMA_7"], label="7-Day SMA", color ='green')
    plt.plot(data.index, data["SMA_30"], label="30-Day SMA", color ='yellow')

    anomalies = data.loc[anomalies_indices]
    plt.scatter(anomalies.index, anomalies["Close"], color="red", label="Anomalies", marker="x")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()