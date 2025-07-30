
---

# 🧠 Crypto Anomaly Detection using Autoencoders and Attention Mechanism

## 📌 Introduction

This project implements a robust anomaly detection pipeline for cryptocurrency price data (specifically **BTC/USDT**) using traditional **Autoencoders**, **Attention-based Autoencoders**, and **Isolation Forest**. It leverages Binance's historical market data and applies advanced preprocessing and feature engineering to build models capable of detecting outliers in market behavior.

## 📚 Table of Contents

* [Introduction](#-introduction)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Dependencies](#dependencies)
* [Configuration](#configuration)
* [Documentation](#documentation)
* [Examples](#examples)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/crypto-anomaly-detection.git
   cd crypto-anomaly-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have TensorFlow and Keras installed:

   ```bash
   pip install tensorflow keras
   ```

---

## 🚀 Usage

Run the project using:

```bash
python main.py
```

This will:

1. Fetch historical BTC/USDT data from Binance.
2. Preprocess and engineer features.
3. Train:

   * An Isolation Forest model.
   * A Dense Autoencoder model.
   * A Self-Attention-based Autoencoder model.
4. Detect and export anomaly results into CSV files.
5. Generate anomaly detection visualizations.

---

## ✨ Features

* 📥 **Live data collection** from Binance.
* 🧪 **Comprehensive preprocessing** including outlier handling, scaling, and log transformations.
* 🔍 **Three anomaly detection models**:

  * Isolation Forest
  * Dense Autoencoder
  * Attention-based Autoencoder
* 📊 **Visualization** of anomalies detected.
* 📁 **Output** stored as:

  * `BTCUSDT_processed_data.csv`
  * `anomalies_isolation_forest.csv`
  * `anomalies_autoencoder.csv`
  * `anomalies_attention.csv`

---

## 🧩 Dependencies

* `tensorflow`
* `keras`
* `numpy`
* `pandas`
* `matplotlib`
* `sklearn`
* `scipy`
* `binance`

---

## ⚙️ Configuration

Update your Binance API keys in `data_collection.py`:

```python
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
```

Set training parameters directly in `main.py` under:

```python
train_autoencoder(...)
train_attention_ae(...)
```

---

## 📖 Documentation

### Key Modules

* `data_collection.py`: Pulls data from Binance.
* `preprocessing.py`: Cleans and computes financial features.
* `feature_engineering.py`: Applies scaling, winsorization, and log transformation.
* `autoencoder.py`: Dense autoencoder model for anomaly detection.
* `attention.py`: Self-attention autoencoder architecture.
* `main.py`: Full end-to-end pipeline.

---

## 🧪 Examples

Here’s how to fetch and preprocess BTCUSDT data:

```python
data = fetch_binance_data("BTCUSDT", "1d", "1 Jan 2018")
data = load_data("BTCUSDT_historical_data.csv")
data = calculate_features(data)
```

To train and use the autoencoder:

```python
autoencoder = build_autoencoder(input_dim=11)
train_autoencoder(autoencoder, X_train)
anomalies, errors = detect_anomalies_autoencoder(autoencoder, X_test)
```

---

## 🛠 Troubleshooting

* **API Rate Limits**: Binance has strict rate limits. Include appropriate `time.sleep()` or retry logic if you face issues.
* **Data Errors**: Ensure all feature columns exist and are properly transformed.
* **Model Convergence**: Adjust learning rate or number of epochs for better results.

---




