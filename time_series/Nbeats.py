import argparse
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Lambda, Subtract, Add, Reshape, Input
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from os.path import join
import os


DATA_PATH="/cs/labs/peleg/tzvi.michelson/time_series/data"
WINDOW_LENGTH = 12
STEP = 2

def build_positional_addings(sample_dim, num_samples):
    """
    Make sure sample_dim is even when running.
    """
    def w(x):
        return 1 / (10000**(2*x/sample_dim))
    positional = []
    for i in range(num_samples):
        pos = []
        for j in range(int(sample_dim/2)):
            pos.append(math.sin(w(j) * (i)))
            pos.append(math.cos(w(j) * (i)))
        pos = np.array(pos)
        positional.append(pos)
    positional = np.array(positional)
    return positional

def window_andPE(X, window_length=WINDOW_LENGTH, step=STEP):
    assert (len(X[0]) - (window_length - step)) % step == 0
    windowed = []
    PE = build_positional_addings(window_length, int((len(X[0]) - (window_length - step))/step) + 1) # added +1 for class encoding
    for x in X:
        current = 0
        single_windowed = []
        while current+window_length <= len(x):
            single_windowed.append(x[current:current+window_length])
            current += step
        single_windowed = np.array(single_windowed)
        # class_token = np.random.randint(np.min(single_windowed), np.max(single_windowed), size=(1, WINDOW_LENGTH)) # add class token
        class_token = np.zeros(shape=(1, WINDOW_LENGTH)) # add class token
        single_windowed = np.concatenate((single_windowed, class_token))
        assert single_windowed.shape == PE.shape
        single_windowed += PE
        windowed.append(single_windowed)
    windowed = np.array(windowed, dtype=np.float32)
    print(windowed.shape)
    return windowed

def smape(y_actual, y_pred):
    s_mape = 200 * K.mean(K.abs(y_actual-y_pred) / (K.abs(y_actual) + K.abs(y_actual)))
    return s_mape

class NBeatsNet:
    GENERIC_BLOCK = 'generic'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'

    _BACKCAST = 'backcast'
    _FORECAST = 'forecast'

    def __init__(self,
                 input_dim=1,
                 exo_dim=0,
                 backcast_length=10,
                 forecast_length=2,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None):

        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.input_dim = input_dim
        self.exo_dim = exo_dim
        self.input_shape = (self.backcast_length, self.input_dim)
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.output_shape = (self.forecast_length, self.input_dim)
        self.weights = {}
        self.nb_harmonics = nb_harmonics
        assert len(self.stack_types) == len(self.thetas_dim)

        x = Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
            x_[k] = Reshape(target_shape=(self.backcast_length, 1))(x_[k])
        if self.input_dim > 1:
            y_ = Concatenate()([y_[ll] for ll in range(self.input_dim)])
            x_ = Concatenate()([x_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]
            x_ = x_[0]

        if self.has_exog():
            n_beats_forecast = Model([x, e], y_, name=self._FORECAST)
            n_beats_backcast = Model([x, e], x_, name=self._BACKCAST)
        else:
            n_beats_forecast = Model(x, y_, name=self._FORECAST)
            n_beats_backcast = Model(x, x_, name=self._BACKCAST)

        self.models = {model.name: model for model in [n_beats_backcast, n_beats_forecast]}
        self.cast_type = self._FORECAST

    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.exo_dim > 0

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model
        return load_model(filepath, custom_objects, compile)

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):
        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}
        d1 = reg(Dense(self.units, activation='relu', name=n('d1')))
        d2 = reg(Dense(self.units, activation='relu', name=n('d2')))
        d3 = reg(Dense(self.units, activation='relu', name=n('d3')))
        d4 = reg(Dense(self.units, activation='relu', name=n('d4')))
        if stack_type == 'generic':
            theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))
        elif stack_type == 'trend':
            theta_f = theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f_b')))
            backcast = Lambda(trend_model, arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
            forecast = Lambda(trend_model, arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = reg(Dense(self.nb_harmonics, activation='linear', use_bias=False, name=n('theta_b')))
            else:
                theta_b = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(seasonality_model,
                              arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            d1_ = d1(d0)
            d2_ = d2(d1_)
            d3_ = d3(d2_)
            d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.n_beats.predict()
        # same for fit(), train_on_batch()...
        attr = getattr(self.models[self._FORECAST], name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            cast_type = self._FORECAST
            if attr.__name__ == 'predict' and 'return_backcast' in kwargs and kwargs['return_backcast']:
                del kwargs['return_backcast']
                cast_type = self._BACKCAST
            return getattr(self.models[cast_type], attr.__name__)(*args, **kwargs)

        return wrapper

def linear_space(backcast_length, forecast_length, is_forecast=True):
    ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    return ls[backcast_length:] if is_forecast else ls[:backcast_length]

def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)

def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)]))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))


def run(X, y, blocks=8, hidden_layer_dim=64):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # input = Input(shape=(13,12))

    Model = NBeatsNet(stack_types=(
        NBeatsNet.GENERIC_BLOCK,
        ),
        nb_blocks_per_stack=blocks,
        forecast_length=1,
        backcast_length=60,
        thetas_dim=(8,), # For each stack we need a dim for it's theta
        share_weights_in_stack=False,  # just change it to True.
        hidden_layer_units=hidden_layer_dim)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)
    Model.compile(loss=smape, optimizer='adam')
    Model.fit(X_train, y_train, batch_size=64, epochs=100, verbose=2, validation_split=0.15, callbacks=[callback])
    print(Model.evaluate(X_test, y_test, verbose=1))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run a transformer to predict time series')
    parser.add_argument('--encoders', type=int, default=2, help='Number of encoders to put in transformer')
    parser.add_argument('--heads', type=int, default=3, help='Number of heads per encoder in the transformer')
    parser.add_argument('--window_length', type=int,  default=12,help='Length of window in data')
    parser.add_argument('--step', type=int,  default=2,help='Size step before windowing (must align with size of input in data and window size)')
    parser.add_argument('--dff', type=int,  default=48, help='Size of hidden layer in transformer feed forward')

    args = vars(parser.parse_args())
    X = np.load(join(DATA_PATH, "X"))
    y = np.load(join(DATA_PATH, "y"))
    print("X shape: ", X.shape)
    print("y shape: ", y.shape)
    for blocks in range(20, 71, 5):
        # for hidden_dim in (30, 60, 120, 180,  240, 360, 480):
        print("Hidden dim: ", 480)
        print("Number of blocks: ", blocks)
        run(X, y, blocks=blocks, hidden_layer_dim=480)

