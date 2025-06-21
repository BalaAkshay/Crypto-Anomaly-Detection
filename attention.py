import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import numpy as np

class SelfAttention(Layer):
    def __init__(self, attention_dim=32, **kwargs):
        self.attention_dim = attention_dim
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', 
                               shape=(input_shape[-1], self.attention_dim),
                               initializer='glorot_uniform')
        self.b = self.add_weight(name='att_bias',
                               shape=(self.attention_dim,),
                               initializer='zeros')
        self.V = self.add_weight(name='att_v',
                               shape=(self.attention_dim, 1),
                               initializer='glorot_uniform')
        super(SelfAttention, self).build(input_shape)
    
    def call(self, x):
        # Calculate attention scores
        score = K.relu(K.dot(x, self.W) + self.b)
        logits = K.dot(score, self.V) / K.sqrt(K.cast(self.attention_dim, dtype='float32'))
        attention_weights = K.softmax(logits, axis=1)

        
        # Apply attention weights
        context_vector = attention_weights * x
        return K.sum(context_vector, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

from tensorflow.keras.layers import Reshape, TimeDistributed

from tensorflow.keras.layers import Reshape, TimeDistributed, Dense, Dropout, BatchNormalization

def build_attention_autoencoder(input_dim, encoding_dim=16, attention_dim=32, time_steps=7):
    feature_dim = input_dim // time_steps
    assert input_dim % time_steps == 0, "Input dimension must be divisible by time_steps"

    input_layer = Input(shape=(input_dim,))

    # Reshape flat input to (time_steps, feature_dim)
    x = Reshape((time_steps, feature_dim))(input_layer)

    # TimeDistributed Encoder (stacked dense layers)
    x = TimeDistributed(Dense(64, activation='relu'))(x)

    x = TimeDistributed(Dense(32, activation='relu'))(x)
    
    x = TimeDistributed(Dense(16, activation='relu'))(x)

    x = TimeDistributed(Dense(encoding_dim, activation='relu'))(x)

    # Attention Layer
    x = SelfAttention(attention_dim=attention_dim)(x)

    # Decoder (dense reconstruction layers)
   
    x = Dense(16, activation='relu')(x)

    x = Dense(32, activation='relu')(x)
    
    x = Dense(64, activation='relu')(x)


    output = Dense(input_dim, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse')

    return model



def train_attention_ae(attention_ae, X_train, epochs = 50, batch_size = 32, validation_data = None):
    
    history = attention_ae.fit(

        X_train, X_train,
        epochs = epochs,
        batch_size = batch_size,
        shuffle = True,
        validation_data = validation_data

    )

    return history


def detect_anomalies_attention_ae(attention_ae, X_test, threshold_percentile = 95):

    X_test_reconstructed = attention_ae.predict(X_test)

    reconstruction_error_attention = np.mean(np.power(X_test - X_test_reconstructed, 2), axis = 1)

    threshold = np.percentile(reconstruction_error_attention, threshold_percentile)

    anomalies_attention = reconstruction_error_attention > threshold

    return anomalies_attention, reconstruction_error_attention
# # Example usage:
# if __name__ == "__main__":
#     # Generate dummy data (replace with your actual data)
#     input_dim = 10
#     X_train = np.random.rand(1000, input_dim)
#     X_test = np.random.rand(200, input_dim)
    
#     # Build and train the attention autoencoder
#     attention_ae = build_attention_autoencoder(input_dim)
#     attention_ae.fit(X_train, X_train,
#                     epochs=50,
#                     batch_size=32,
#                     validation_data=(X_test, X_test))
    
#     # Get attention weights for interpretation
#     attention_layer = attention_ae.layers[2]  # SelfAttention layer
#     get_attention = K.function([attention_ae.input], 
#                               [attention_layer.output])
#     attention_weights = get_attention([X_test[:1]])[0]
#     print("Attention weights for first sample:", attention_weights)