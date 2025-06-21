import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam


def build_autoencoder(input_dim, encoding_dim=32):

    autoencoder = Sequential([
        
        Input(shape=(input_dim,)),
        Dense(64, activation = 'relu'),
        Dense(32, activation = 'relu'),
        Dense(16, activation = 'relu'),

        Dense(encoding_dim, activation = 'relu'),
        
        Dense(16, activation = 'relu'),
        Dense(32, activation = 'relu'),
        Dense(64, activation = 'relu'),

        Dense(input_dim, activation = 'linear')
    
    ])

    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return autoencoder


def train_autoencoder(autoencoder, X_train, epochs = 50, batch_size = 32, validation_data = None):
    
    history = autoencoder.fit(

        X_train, X_train,
        epochs = epochs,
        batch_size = batch_size,
        shuffle = True,
        validation_data = validation_data

    )

    return history


def detect_anomalies_autoencoder(autoencoder, X_test, threshold_percentile = 95):

    X_test_reconstructed = autoencoder.predict(X_test)

    reconstruction_error = np.mean(np.power(X_test - X_test_reconstructed, 2), axis = 1)

    threshold = np.percentile(reconstruction_error, threshold_percentile)

    anomalies_auto = reconstruction_error > threshold

    return anomalies_auto, reconstruction_error
