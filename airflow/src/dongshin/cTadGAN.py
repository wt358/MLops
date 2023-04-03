import tensorflow as tf
import keras
#import similaritymeasures as sm
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Bidirectional, LSTM, Flatten, Dense, Reshape, UpSampling1D, TimeDistributed
from tensorflow.keras.layers import Activation, Conv1D, LeakyReLU, Dropout, Add, Layer
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.optimizers import Adam

from functools import partial
from scipy import integrate, stats

class RandomWeightedAverage(Layer):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
    
def build_encoder_layer(input_shape, encoder_reshape_shape):
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
        Bidirectional(LSTM(units=100, return_sequences=True)),
        Flatten(),
        Dense(20),
        Reshape(target_shape=encoder_reshape_shape)])  # (20, 1)

    return Model(x, model(x))


def build_generator_layer(input_shape, generator_reshape_shape):
    # input_shape = (20, 1) / generator_reshape_shape = (50, 1)
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
        Flatten(),
        Dense(50),
        Reshape(target_shape=generator_reshape_shape),  # (50, 1)
        Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat'),
        Dropout(rate=0.2),
        UpSampling1D(size=2),
        Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat'),
        Dropout(rate=0.2),
        TimeDistributed(Dense(1)),
        Activation(activation='tanh')])  # (None, 100, 1)

    return Model(x, model(x))


def build_critic_x_layer(input_shape):
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
        Conv1D(filters=64, kernel_size=5),
        LeakyReLU(alpha=0.2),
        Dropout(rate=0.25),
        Conv1D(filters=64, kernel_size=5),
        LeakyReLU(alpha=0.2),
        Dropout(rate=0.25),
        Conv1D(filters=64, kernel_size=5),
        LeakyReLU(alpha=0.2),
        Dropout(rate=0.25),
        Conv1D(filters=64, kernel_size=5),
        LeakyReLU(alpha=0.2),
        Dropout(rate=0.25),
        Flatten(),
        Dense(units=1)])

    return Model(x, model(x))


def build_critic_z_layer(input_shape):
    x = Input(shape=input_shape)
    model = tf.keras.models.Sequential([
        Flatten(),
        Dense(units=100),
        LeakyReLU(alpha=0.2),
        Dropout(rate=0.2),
        Dense(units=100),
        LeakyReLU(alpha=0.2),
        Dropout(rate=0.2),
        Dense(units=1)])

    return Model(x, model(x))

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


