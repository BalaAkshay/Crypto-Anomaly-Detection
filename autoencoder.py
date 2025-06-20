import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# === 1. Build Improved Autoencoder ===
def build_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = Dense(128)(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    encoded = Dense(encoding_dim, activation='relu')(x)

    # Decoder
    x = Dense(64)(encoded)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    decoded = Dense(input_dim, activation='sigmoid')(x)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return autoencoder


# === 2. Train Autoencoder with EarlyStopping ===
def train_autoencoder(autoencoder, X_train, epochs=100, batch_size=32, validation_data=None):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=validation_data,
        callbacks=[early_stop] if validation_data else None
    )

    return history


# === 3. Detect Anomalies and Label Pump/Dump ===
def detect_anomalies_autoencoder(autoencoder, X_test, threshold_percentile=95, return_threshold=False):
    X_test_reconstructed = autoencoder.predict(X_test, verbose=0)

    reconstruction_error = np.mean(np.square(X_test - X_test_reconstructed), axis=1)
    threshold = np.percentile(reconstruction_error, threshold_percentile)

    anomalies = reconstruction_error > threshold

    # Classify pump or dump based on 'Close' direction
    close_prices = X_test[:, 0]  # assuming 'Close' is the first feature
    daily_returns = X_test[:, 1]  # assuming 'Daily Return' is the second feature

    labels = np.array(['Normal'] * len(X_test))
    labels[anomalies & (daily_returns > 0)] = 'Pump'
    labels[anomalies & (daily_returns <= 0)] = 'Dump'

    if return_threshold:
        return anomalies, reconstruction_error, threshold, labels
    else:
        return anomalies, reconstruction_error, labels
