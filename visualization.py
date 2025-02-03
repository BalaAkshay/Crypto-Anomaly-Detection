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