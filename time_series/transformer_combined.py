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
        if sample_dim % 2 == 1:
            pos.append(math.sin(w(j+1) * (i)))
        pos = np.array(pos)
        positional.append(pos)
    print("j in positional encoding is: ", j)
    positional = np.array(positional)
    return positional

def window_andPE(X, window_length=WINDOW_LENGTH, step=STEP, do_pe=False, add_class_token=True):
    # print("(len(X[0]) - (window_length - step)) % step: ", (len(X[0]) - (window_length - step)) % step)
    assert (len(X[0]) - (window_length - step)) % step == 0
    windowed = []
    if add_class_token:
        PE = build_positional_addings(window_length, int((len(X[0]) - (window_length - step))/step) + 1) # added +1 for class encoding
    else:
        PE = build_positional_addings(window_length, int((len(X[0]) - (window_length - step))/step)) # didnt add +1 for class encoding
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
    s_mape = 200 * K.mean(K.abs(y_actual-y_pred) / (K.abs(y_actual) + K.abs(y_pred)))
    return s_mape

def smape_numpy(y_actual, y_pred):
    s_mape = 200 * np.mean(np.abs(y_actual-y_pred) / (np.abs(y_actual) + np.abs(y_pred)))
    return s_mape

def point_wise_feed_forward_network(d_model, dff):
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

        self.wq = Dense(d_model)#, kernel_initializer=tf.keras.initializers.Identity())
        self.wk = Dense(d_model)#, kernel_initializer=tf.keras.initializers.Identity())
        self.wv = Dense(d_model, kernel_initializer=tf.keras.initializers.Zeros())#, kernel_initializer=tf.keras.initializers.Identity())

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
    def __init__(self, d_model, num_heads, token_input, dff, rate, connecting_tokens, norm):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.token_input = token_input
        self.connecting_tokens = connecting_tokens
        self.norm = norm
        # self.dense1 = Dense(d_model)
        self.mha = MultiHeadAttention(token_input, num_heads)
        # self.dense2 = Dense(token_input)
        # probelmatic!!!! magic number
        if connecting_tokens:
            self.ffn = point_wise_feed_forward_network(d_model, dff)
        else:
            self.ffn = point_wise_feed_forward_network(token_input, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # attn_input = self.dense1(x)
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        # attn_output = self.dense2(attn_output)

        attn_output = self.dropout1(attn_output, training=training)

        if self.norm:
            out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out1 = x + attn_output

        if self.connecting_tokens:
            out_reshaped = tf.reshape(out1, (out1.shape[0], out1.shape[1] * out1.shape[2]))
            ffn_output = self.ffn(out_reshaped)  # (batch_size, input_seq_len, d_model)
            ffn_output = tf.reshape(ffn_output, (ffn_output.shape[0], -1, self.token_input))
        else:
            ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.dropout2(ffn_output, training=training)

        if self.norm:
            out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out2 = out1 + ffn_output


        return out2

class Transformer(tf.keras.Model):
    def __init__(self, encoders, d_model, token_input, num_heads, dff, rate, connecting_tokens, norm):
        """
        encoders: number of encoders to put in
        d_model: total dimension for attention: output_dim * number of heads
        num_heads: number of attention heads per encoder layer
        dff: dimensioality of non-linear dense layer that is applied pointwise.
        """
        super(Transformer, self).__init__()
        self.encoders = [EncoderLayer(d_model, num_heads, token_input, dff, rate, connecting_tokens, norm) for _ in range(encoders)]
        self.flatten = Flatten()
        self.dense = Dense(1, activation='linear')

    def call(self, x, training):
        for encoder in self.encoders:
            # print("calling 1")
            x = encoder(x, training, None)
            # print("called 1")
        flat = self.flatten(x)
        # last = x[:,-1:,:]
        # print(last.shape)
        # flat = self.flatten(last)
        # print(flat.shape)
        out = self.dense(flat)
        return out

def run(X, y, lr=0.001, encoders=2, heads=2, dff=None, window_length=WINDOW_LENGTH, step=STEP, d_model=60,  do_pe=False, add_class_token=False, connecting_tokens=True, norm=True):
    dff = 4 * d_model
    windowed = window_andPE(X, window_length=window_length, step=step, do_pe=do_pe, add_class_token=add_class_token)
    X_train, X_test, y_train, y_test = train_test_split(windowed, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


    # input = Input(shape=(13,12))

    model = Transformer(encoders=encoders, d_model=d_model, token_input=window_length, num_heads=heads, dff=dff, rate=0, connecting_tokens=connecting_tokens, norm=norm)

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
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(128)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    EPOCHS = 500
    # model.load_weights(MODEL_PATH + "signal_to_signal_sdr")
    # Training
    print("l"
          "\n\nLearning rate is: ", lr)
    for epoch in range(1, EPOCHS + 1):
        train_loss.reset_states()
        test_loss.reset_states()
        for inp, out in train_ds:
            train_step(inp, out)
        for inp, out in val_ds:
            test_step(inp, out)
        if epoch % 2 == 0:
            print(
                f'Epoch {epoch}, '
                f'Loss: {train_loss.result()}, '
                f'Test Loss: {test_loss.result()}, '
            )
            if epoch % 5 == 0:
                curr = test_loss.result()
                if epoch > 20 and curr > prev:
                    break
                prev = curr

    predictions = []
    truth = []
    for input, output in test_ds:
        results = model(input, training=False).numpy()
        for result in results:
            predictions.append(result)
        for a in output:
            truth.append(a)

    predictions = np.array(predictions)
    truth = np.array(truth)
    print("Smape on test set is: ", smape_numpy(truth, predictions))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run a transformer to predict time series')
    parser.add_argument('--encoders', type=int, default=2, help='Number of encoders to put in transformer')
    parser.add_argument('--heads', type=int, default=3, help='Number of heads per encoder in the transformer')
    parser.add_argument('--window_length', type=int, default=12, help='Length of window in data')
    parser.add_argument('--step', type=int,  default=2, help='Size step before windowing (must align with size of input in data and window size)')
    parser.add_argument('--dff', type=int,  default=48, help='Size of hidden layer in transformer feed forward')

    args = vars(parser.parse_args())
    X = np.load(join(DATA_PATH, "X"))
    y = np.load(join(DATA_PATH, "y"))
    SAMPLE_SIZE = X.shape[1]
    assert SAMPLE_SIZE == 60
    # for a in args.keys():
    #     print(a, ": ", args[a])
    # run(X, y, encoders=args["encoders"], heads=args["heads"], dff=args["dff"] ,window_length=args["window_length"], step=args["step"])
    Window_length = 30
    Step = 30
    learning_rate = 0.001
    lst = [(1, 1, 30, 30, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 1, 15, 15, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 1, 12, 12, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 1, 10, 10, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 2, 30, 30, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 3, 30, 30, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 3, 15, 15, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 2, 12, 12, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 3, 12, 12, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True),
           (1, 2, 10, 10, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True)]

    lst = [(1, 1, 30, 30, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True, True),
           (1, 1, 15, 15, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True, True),
           (1, 1, 12, 12, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True, True),
           (1, 1, 10, 10, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True, True),
           (1, 1, 6, 6, SAMPLE_SIZE*4, learning_rate, SAMPLE_SIZE, True, True)]

    for encoders, heads, window_length, step, dff, lr, d_model, connecting_tokens, norm in lst:
        print("\n\nWindow length: ", window_length)
        print("Heads: ", heads)
        print("Encoders: ", encoders)
        print("connecting tokens = ", connecting_tokens)
        run(X, y, lr=lr, encoders=encoders, heads=heads, dff=dff ,window_length=window_length, d_model=d_model, step=step, do_pe=True, add_class_token=False, connecting_tokens=connecting_tokens, norm=norm)

# 30 2 (1 head) identity initialization - train 46.8 test 51.1
# 30 2 (1 head) default initialization - train 46 test 51
# 30 2 (1 head) zero initialization - increase dim - train 41 test 50
# 30 2 (1 head) zero initialization - train 37 test 46.9
# 30 2 (1 head) zero initialization with PE - train 37 test 46.5
# 30 2 (1 head) zero initialization with PE and without reconnecting for dff - train 39 test 48
# 30 2 (1 head) zero initialization with PE and with reconnecting with norm - train 35.08 test 44.90
# 30 2 (1 head) zero initialization with PE and without reconnecting with norm - train 42.49 test 47.14

# 30 2 (1 head) zero initialization with PE and with reconnecting with norm - train 36.4 test 44.6
# 15 4 (1 head) zero initialization with PE and with reconnecting with norm - train 35.11 test 44.33
# 12 5 (1 head) zero initialization with PE and with reconnecting with norm - train 34.51 test 43.98
# 10 6 (1 head) zero initialization with PE and with reconnecting with norm - train 37.61 test 46.01
# 6 12 (1 head) zero initialization with PE and with reconnecting with norm - train 38.66 test 46.52

# 12 5 (1 head) - 2 encoders - zero initialization with PE and with reconnecting with norm - train 28.20 test 42.73
# 12 5 (1 head) - 3 encoders - zero initialization with PE and with reconnecting with norm - train 29.16 test 43.04
# 12 5 (1 head) - 4 encoders - zero initialization with PE and with reconnecting with norm - train 63.46 test 63.09
# 12 5 (1 head) - 5 encoders - zero initialization with PE and with reconnecting with norm - train 63.46 test 63.09
# 12 5 (1 head) - 6 encoders - zero initialization with PE and with reconnecting with norm - train 63.46 test 63.09

# 12 5 (1 head) - 2 encoders - zero initialization with PE and without reconnecting with norm with adding outputs - train 49.26 test 51.49
# 12 5 (1 head) - 4 encoders - zero initialization with PE and without reconnecting with norm with adding outputs - train 48.65 test 51.68
# 12 5 (1 head) - 6 encoders - zero initialization with PE and without reconnecting with norm with adding outputs - train 62.93 test 62
# 12 5 (1 head) - 8 encoders - zero initialization with PE and without reconnecting with norm with adding outputs - train 61.77 test 61.08

# 12 5 (1 head) - 2 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 27.89 test 42.72
# 12 5 (1 head) - 4 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 31.13 test 42.92
# 12 5 (1 head) - 6 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 30.03 test 42.3
# 12 5 (1 head) - 8 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 29.91 test 44.06
# 12 5 (1 head) - 10 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 31.79 test 42.96
# 12 5 (1 head) - 12 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 31.12 test 43.04
# 12 5 (1 head) - 14 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 32.73 test 43.65
# 12 5 (1 head) - 16 encoders - zero initialization with PE and with reconnecting with norm with adding outputs - train 34.11 test 43.8

# 12 5 (1 head) - 2 encoders - zero initialization with PE and with reconnecting with norm with adding outputs with subtraction - train 34.99 test 43.05
# 12 5 (1 head) - 4 encoders - zero initialization with PE and with reconnecting with norm with adding outputs with subtraction - train 28.61 test 41.46
# 12 5 (1 head) - 6 encoders - zero initialization with PE and with reconnecting with norm with adding outputs with subtraction - train 28.34 test 41.74
# 12 5 (1 head) - 8 encoders - zero initialization with PE and with reconnecting with norm with adding outputs with subtraction - train 30.67 test 42.14
# 12 5 (1 head) - 10 encoders - zero initialization with PE and with reconnecting with norm with adding outputs with subtraction - train 29.94 test 42.37

# Nbeats: 8 blocks 1 stack 64 hidden unit

# Explanation: with reconnecting = reconnect tokens of transformer before mlp
# with adding outputs = with adding the outputs after every two encoders of the transformer and giving at the end the sum of all outputs
# with subtraction means subtracting from what we have before the two encoders a linear projection of what we got after the two encoders

# 30 2 (2 heads) gives train - 45 test - 38
# 30 2 gives test - 43(45)
# 12 5 gives train - 57 test - 57



