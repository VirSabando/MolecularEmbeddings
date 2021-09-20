import tensorflow as tf
from tensorflow import keras
from my_layers import SelfAttentiveLayer
from keras.regularizers import l2

if tf.test.is_gpu_available():
    LSTM = tf.compat.v1.keras.layers.CuDNNLSTM
else:
    LSTM = tf.compat.v1.keras.layers.LSTM

dropout_coefficient = 0.2
lambda_ln = 0.01
regularizer = l2(lambda_ln)

# Using notation stated in the paper by Zheng et al
# n: number of tokens in each SMILES (after padding). It is THE SAME for all SMILES
# d: dimensionality of each token vector computed using mol2vec. Using the pretrained model, d=300
# u: number of hidden units per LSTM nn. In a Bidirectional LSTM, the final number of hidden units would be 2u
# da: number of parameters of weight vectors W1 and w2, as stated in the paper
# r: number of heads in the multi-head attention, also called number of attention rows
def build_sa_bilstm_model(n, d, u, da, r):
    inputs = keras.layers.Input(shape=(n, d), dtype='float32')
    # Add the merge mode -->> merge_mode='concat'?
    sequence = keras.layers.Bidirectional(
        LSTM(u, return_sequences=True, kernel_regularizer=regularizer))(inputs)  
    drop = keras.layers.Dropout(dropout_coefficient)(sequence)
    self_attention = SelfAttentiveLayer(r, da, 2*u)(drop)
    flatten = keras.layers.Flatten()(self_attention)
    y = keras.layers.Dense(1, activation = 'sigmoid')(flatten)
    # may be activation could be added here
    model = keras.models.Model(inputs=inputs, outputs=y)
    return model
