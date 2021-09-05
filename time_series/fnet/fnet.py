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

def window_andPE(X, window_length=WINDOW_LENGTH, step=STEP, do_pe=False, add_class_token=True):
    # print("(len(X[0]) - (window_length - step)) % step: ", (len(X[0]) - (window_length - step)) % step)
    assert (len(X[0]) - (window_length - step)) % step == 0
    windowed = []
    if add_class_token:
        PE = build_positional_addings(window_length, int((len(X[0]) - (window_length - step))/step) + 1) # added +1 for class encoding
    else:
        PE = build_positional_addings(window_length, int((len(X[0]) - (window_length - step))/step)) # added +1 for class encoding
    for x in X:
        current = 0
        single_windowed = []
        while current+window_length <= len(x):
            single_windowed.append(x[current:current+window_length])
            current += step
        single_windowed = np.array(single_windowed)
        # class_token = np.random.randint(np.min(single_windowed), np.max(single_windowed), size=(1, window_length)) # add class token
        class_token = np.zeros(shape=(1, window_length)) # add class token
        if add_class_token:
            single_windowed = np.concatenate((single_windowed, class_token))
        assert single_windowed.shape == PE.shape
        if do_pe:
            single_windowed += PE
        windowed.append(single_windowed)
    windowed = np.array(windowed, dtype=np.float32)
    # print(windowed.shape)
    return windowed

def smape(y_actual, y_pred):
    s_mape = 200 * K.mean(K.abs(y_actual-y_pred) / (K.abs(y_actual) + K.abs(y_actual)))
    return s_mape

def point_wise_feed_forward_network(d_model, dff, first):
    if first:
        return tf.keras.Sequential([
            Dense(dff, activation='relu', kernel_initializer=tf.keras.initializers.Zeros()),  # (batch_size, seq_len, dff)
            Dense(d_model)  # (batch_size, seq_len, d_model)
        ])
    else:
        return tf.keras.Sequential([
            Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # print("d_model % self.num_heads: ", d_model % self.num_heads)
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # print(concat_attention.shape)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        # print(output.shape)
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, first=False, rate=0.1):
        super(EncoderLayer, self).__init__()

        # self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff, first)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # print("calling 2")

        # attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        x_complex = tf.cast(x, tf.complex64)
        attn_output = tf.signal.fft2d(x_complex)
        attn_output = tf.abs(attn_output)
        # print("calling 3")

        attn_output = self.dropout1(attn_output, training=training)
        # print("calling 4")
        # out1 = x + attn_output
        # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        # print("calling 5")

        ffn_output = self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        out2 = x + ffn_output

        return out2

class Transformer(tf.keras.Model):
    def __init__(self, encoders, d_model, big_dim, token_input, num_heads, dff, rate, connecting_tokens, norm):
        """
        encoders: number of encoders to put in
        d_model: total dimension for attention: output_dim * number of heads
        num_heads: number of attention heads per encoder layer
        dff: dimensioality of non-linear dense layer that is applied pointwise.
        """
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.encoders = [EncoderLayer(big_dim, d_model, num_heads, token_input, dff) for _ in range(encoders)]
        self.flatten = Flatten()
        self.denses = [Dense(token_input) for _ in range(int(encoders/2))]
        self.res_sub = [Dense(big_dim) for _ in range(int(encoders/2))]
        self.biggen = Dense(big_dim, activation='linear')
        self.dense = Dense(token_input, activation='linear')

    def call(self, x, training):
        x = self.biggen(x)
        out = None
        block_input = x
        for i in range(len(self.encoders)):
            if (i - 1) % 4 == 0:
                x = self.encoders[i](block_input, training, None)
            else:
                x = self.encoders[i](x, training, None)
            if i % 4 == 0:
                if out is None:
                    flat = self.flatten(x)
                    out  = self.denses[int(i/2)](flat)
                else:
                    flat = self.flatten(x)
                    out  += self.denses[int(i/2)](flat)
                to_sub = self.res_sub[int(i/2)](x)
                block_input = block_input - to_sub
        return out

def run(X, y, encoders=2, heads=2, dff=4*WINDOW_LENGTH ,window_length=WINDOW_LENGTH, step=STEP, do_pe=False, add_class_token=False):

    windowed = window_andPE(X, window_length=window_length, step=step, do_pe=do_pe, add_class_token=add_class_token)
    X_train, X_test, y_train, y_test = train_test_split(windowed, y, test_size=0.1, random_state=42)

    # input = Input(shape=(13,12))

    model = Transformer(encoders=encoders, d_model=window_length, num_heads=heads, dff=dff, rate=0)
    # print("created model")
    loss_function = smape
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(inp, out):
        with tf.GradientTape() as tape:
            predictions = model(inp, training=True)
            loss = loss_function(out, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    @tf.function
    def test_step(inp, out):
        predictions = model(inp, training=False)
        t_loss = loss_function(out, predictions)
        test_loss(t_loss)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).batch(128)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128)
    optimizer = tf.keras.optimizers.Adam()

    EPOCHS = 500
    # model.load_weights(MODEL_PATH + "signal_to_signal_sdr")
    # Training
    for epoch in range(1, EPOCHS + 1):
        train_loss.reset_states()
        test_loss.reset_states()
        for inp, out in train_ds:
            train_step(inp, out)
        for inp, out in test_ds:
            test_step(inp, out)
        if epoch % 2 == 0:
            print(
                f'Epoch {epoch}, '
                f'Loss: {train_loss.result()}, '
                f'Test Loss: {test_loss.result()}, '
            )
            if epoch % 30 == 0:
                curr = test_loss.result()
                if epoch > 45 and curr > prev:
                    break
                prev = curr


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run a transformer to predict time series')
    parser.add_argument('--encoders', type=int, default=2, help='Number of encoders to put in transformer')
    parser.add_argument('--heads', type=int, default=3, help='Number of heads per encoder in the transformer')
    parser.add_argument('--window_length', type=int, default=12, help='Length of window in data')
    parser.add_argument('--step', type=int,  default=2, help='Size step before windowing (must align with size of input in data and window size)')
    parser.add_argument('--dff', type=int,  default=48, help='Size of hidden layer in transformer feed forward')

    args = vars(parser.parse_args())
    X = np.load(join("X"))
    y = np.load(join("y"))
    # for a in args.keys():
    #     print(a, ": ", args[a])
    # run(X, y, encoders=args["encoders"], heads=args["heads"], dff=args["dff"] ,window_length=args["window_length"], step=args["step"])
    Window_length = 12
    Step=2
    for encoders, heads, window_length, step, dff in [(4, 1, Window_length, Step, Window_length*4)]:# [(1, 4, 12, 12, 48), (4, 4, 12, 12, 48), (1, 3, 30, 30, 120), (4, 3, 30, 30, 120)]:
        try:
            print("No layer norm!")
            run(X, y, encoders=encoders, heads=heads, dff=dff ,window_length=window_length, step=step, do_pe=False, add_class_token=False)
        except:
            print("ERROR!!!")
            pass
# got to 57 with window size 60
# got to 56 with size 12*5
# got to 58 with 10*6



