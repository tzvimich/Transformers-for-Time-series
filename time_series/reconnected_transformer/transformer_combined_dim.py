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
    def __init__(self, big_dim, d_model, num_heads, token_input, dff, rate, connecting_tokens, norm):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.big_dim = big_dim
        self.token_input = token_input
        self.connecting_tokens = connecting_tokens
        self.norm = norm
        # self.dense1 = Dense(d_model)
        self.mha = MultiHeadAttention(big_dim, num_heads)
        # self.dense2 = Dense(token_input)
        # probelmatic!!!! magic number
        if connecting_tokens:
            self.ffn = point_wise_feed_forward_network(d_model, dff)
        else:
            self.ffn = point_wise_feed_forward_network(big_dim, dff)

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
            ffn_output = tf.reshape(ffn_output, (ffn_output.shape[0], -1, self.big_dim))
        else:
            ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.dropout2(ffn_output, training=training)

        if self.norm:
            out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out2 = out1 + ffn_output

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
        self.encoders = [EncoderLayer(big_dim, d_model, num_heads, token_input, dff, rate, connecting_tokens, norm) for _ in range(encoders)]
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
            if (i - 1) % 2 == 0:
                x = self.encoders[i](block_input, training, None)
            else:
                x = self.encoders[i](x, training, None)
            if i % 2 == 0:
                if out is None:
                    flat = self.flatten(x)
                    out  = self.denses[int(i/2)](flat)
                else:
                    flat = self.flatten(x)
                    out  += self.denses[int(i/2)](flat)
                to_sub = self.res_sub[int(i/2)](x)
                block_input = block_input - to_sub
        return out

def run(X_train, y_train, X_test, y_test, lr=0.001, encoders=2, heads=2, dff=None, window_length=WINDOW_LENGTH, step=STEP, d_model=60, big_dim=64, do_pe=False, add_class_token=False, connecting_tokens=True, norm=True):
    dff = 4 * d_model
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_train = window_andPE(X_train, window_length=window_length, step=step, do_pe=do_pe, add_class_token=add_class_token)
    X_test = window_andPE(X_test, window_length=window_length, step=step, do_pe=do_pe, add_class_token=add_class_token)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    model = Transformer(encoders=encoders, d_model=d_model, big_dim=big_dim, token_input=window_length, num_heads=heads, dff=dff, rate=0, connecting_tokens=connecting_tokens, norm=norm)

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
    print(X_test.shape)
    print(X_test.dtype)
    print(y_test.shape)
    print(y_test.dtype)
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
            if epoch % 10 == 0:
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

def load_data_separate(data_type):
    if data_type == "monthly":
        mat = pd.read_csv(join(DATA_PATH, "Monthly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Monthly-test.csv"))
        horizon = 18
    if data_type == "yearly":
        mat = pd.read_csv(join(DATA_PATH, "Yearly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Yearly-test.csv"))
        horizon = 6
    if data_type == "quarterly":
        mat = pd.read_csv(join(DATA_PATH, "Quarterly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Quarterly-test.csv"))
        horizon = 8
    if data_type == "weekly":
        mat = pd.read_csv(join(DATA_PATH, "Weekly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Weekly-test.csv"))
        horizon = 13
    if data_type == "daily":
        mat = pd.read_csv(join(DATA_PATH, "Daily-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Daily-test.csv"))
        horizon = 14
    if data_type == "hourly":
        mat = pd.read_csv(join(DATA_PATH, "Hourly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Hourly-test.csv"))
        horizon = 48
    X_CYCLES = 4
    Y_CYCLES = 1

    X = []
    y = []
    for column in mat:
        X_maybe = []
        y_maybe = []
        xing = True
        ying = False
        for i in mat[column]:
            if isinstance(i, float) and not math.isnan(i):
                if xing:
                    X_maybe.append(i)
                if ying:
                    y_maybe.append(i)
                if len(X_maybe) == X_CYCLES*horizon and xing:
                    ying = True
                    xing = False
                if len(y_maybe) == Y_CYCLES*horizon and ying:
                    X.append(X_maybe)
                    y.append(y_maybe)
                    X_maybe = []
                    y_maybe = []
                    xing = True
                    ying = False
    X_train = np.array(X)
    y_train = np.array(y)
    m = mat.values[:,1:]
    m = np.array(m)
    X_test = []
    exist = []
    for i in range(len(m)):
        x = m[i].astype(np.float)
        x = x[np.logical_not(np.isnan(x))]
        x = x[-X_CYCLES*horizon:]
        if len(x) == X_CYCLES*horizon:
            X_test.append(x)
            exist.append(i)
    X_test = np.array(X_test)
    y_test = np.array(mat_test.values[:,1:])
    y_test = y_test[exist]
    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)
    assert X_train.shape[1] == X_CYCLES*horizon
    assert y_train.shape[1] == horizon
    assert X_test.shape[1] == X_CYCLES*horizon
    assert y_test.shape[1] == horizon
    assert np.sum(np.isnan(X_test.astype(np.float))) == 0
    assert np.sum(np.isnan(y_test.astype(np.float))) == 0
    assert np.sum(np.isnan(X_train.astype(np.float))) == 0
    assert np.sum(np.isnan(y_train.astype(np.float))) == 0
    return X_train, y_train, X_test, y_test, horizon


def load_data(data_type):

    if data_type == "monthly":
        mat = pd.read_csv(join(DATA_PATH, "Monthly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Monthly-test.csv"))
        horizon = 18
    if data_type == "yearly":
        mat = pd.read_csv(join(DATA_PATH, "Yearly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Yearly-test.csv"))
        horizon = 6
    if data_type == "quarterly":
        mat = pd.read_csv(join(DATA_PATH, "Quarterly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Quarterly-test.csv"))
        horizon = 8
    if data_type == "weekly":
        mat = pd.read_csv(join(DATA_PATH, "Weekly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Weekly-test.csv"))
        horizon = 13
    if data_type == "daily":
        mat = pd.read_csv(join(DATA_PATH, "Daily-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Daily-test.csv"))
        horizon = 14
    if data_type == "hourly":
        mat = pd.read_csv(join(DATA_PATH, "Hourly-train.csv"))
        mat_test = pd.read_csv(join(DATA_PATH, "Hourly-test.csv"))
        horizon = 48
    X_CYCLES = 4
    Y_CYCLES = 1

    X = []
    y = []
    m = mat.values[:,1:]
    for column in m:
        X_maybe = []
        y_maybe = []
        xing = True
        ying = False
        i = 0
        while i < len(column):
            k = column[i]
            if isinstance(k, float) and not math.isnan(k):
                if xing:
                    X_maybe.append(k)
                    i += 1
                if ying:
                    y_maybe.append(k)
                    i += 1
                if len(X_maybe) == X_CYCLES*horizon and xing:
                    ying = True
                    xing = False
                if len(y_maybe) == Y_CYCLES*horizon and ying:
                    X.append(X_maybe)
                    y.append(y_maybe)
                    X_maybe = []
                    y_maybe = []
                    xing = True
                    ying = False
                    i = i - (X_CYCLES*horizon + Y_CYCLES*horizon) + 1
    X_train = np.array(X)
    y_train = np.array(y)
    m = mat.values[:,1:]
    m = np.array(m)
    X_test = []
    exist = []
    for i in range(len(m)):
        x = m[i].astype(np.float)
        x = x[np.logical_not(np.isnan(x))]
        x = x[-X_CYCLES*horizon:]
        if len(x) == X_CYCLES*horizon:
            X_test.append(x)
            exist.append(i)
    X_test = np.array(X_test)
    y_test = np.array(mat_test.values[:,1:])
    y_test = y_test[exist]
    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)
    assert X_train.shape[1] == X_CYCLES*horizon
    assert y_train.shape[1] == horizon
    assert X_test.shape[1] == X_CYCLES*horizon
    assert y_test.shape[1] == horizon
    assert np.sum(np.isnan(X_test.astype(np.float))) == 0
    assert np.sum(np.isnan(y_test.astype(np.float))) == 0
    assert np.sum(np.isnan(X_train.astype(np.float))) == 0
    assert np.sum(np.isnan(y_train.astype(np.float))) == 0
    return X_train, y_train, X_test, y_test, horizon


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run a transformer to predict time series')
    parser.add_argument('--encoders', type=int, default=2, help='Number of encoders to put in transformer')
    parser.add_argument('--heads', type=int, default=3, help='Number of heads per encoder in the transformer')
    parser.add_argument('--window_length', type=int, default=12, help='Length of window in data')
    parser.add_argument('--step', type=int,  default=2, help='Size step before windowing (must align with size of input in data and window size)')
    parser.add_argument('--dff', type=int,  default=48, help='Size of hidden layer in transformer feed forward')

    args = vars(parser.parse_args())

    X_train, y_train, X_test, y_test, horizon = load_data_separate("monthly")
    SAMPLE_SIZE = X_train.shape[1]

    for big_dim in [32, 64, 128, 256, 512]:
        attention_heads=2
        print("\nBig dim is: ", big_dim)
        print("attention_heads: ", attention_heads)
        step = horizon
        run(X_train, y_train, X_test, y_test, lr=0.001, encoders=8, heads=attention_heads, dff=None ,window_length=horizon, d_model=int(big_dim*(SAMPLE_SIZE / horizon)), big_dim=big_dim, step=horizon, do_pe=True, add_class_token=False, connecting_tokens=True, norm=True)



