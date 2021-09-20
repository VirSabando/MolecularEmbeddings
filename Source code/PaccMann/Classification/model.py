import tensorflow as tf
from tensorflow import keras
from paccmann_sa import Sequence_Attention_Layer
from paccmann_positional_encoding import sinusoidal_positional_encoding

if tf.test.is_gpu_available():
    LSTM = tf.compat.v1.keras.layers.CuDNNLSTM
else:
    LSTM = tf.compat.v1.keras.layers.LSTM

momentum_batch_norm = 0.95 

# Notation
# n: number of tokens in each SMILES (after padding). It is THE SAME for all SMILES
# d: dimensionality of each token vector.
# u: number of hidden units 
# da: attention depth - used to compute W, B and U vectors
def build_model(params):
    n = params.get('n', 240) # sequence length after padding
    t = params.get('t', 263) # vocabulary size
    
    u = params.get('u', 16) # number of hidden units == embedding size
    da = params.get('da', 256) # attention depth
    act = params.get('activation', 'sigmoid') # activation fn for dense layers
    dropout_coefficient = params.get('dc', 0.5) # dropout coefficient
    l2_lambda = params.get('l', 0.005) # lambda for l2 regularization
    pos_enc = params.get('positional_encoding', False) # positional encoding
    n_layers = params.get('n_layers', 3) # assume 3 layers by default
    dense_sizes_lst = params.get('dense_sizes', [256, 64, 16]) # No. of hidden units per dense layer in the classifier 
                                                                    
    ln_reg = tf.keras.regularizers.l2(l2_lambda)
    
    # Input layer (tokenized sequence)
    inputs = keras.layers.Input(shape=(n), dtype='float32')
    embeddings = keras.layers.Embedding(t, u)(inputs)
    # Positional encoding
    if pos_enc:
        positional_encoding = sinusoidal_positional_encoding(n,u,name='positional_encoding')
        # Implement additive positional encodings.
        encoded_inputs = tf.add(embeddings, positional_encoding,name='additive_positional_embedding')
        sequence = encoded_inputs
    else:
        sequence = embeddings
        
    # Seq_self_attention
    self_attention, alphas = Sequence_Attention_Layer(da, u)(sequence) 
    
    # Batch normalization
    batch_norm = keras.layers.BatchNormalization(momentum=momentum_batch_norm)(self_attention)
    
    # Stack of dense layers for property prediction
    flatten = keras.layers.Flatten()(batch_norm)
    dense_a = keras.layers.Dense(dense_sizes_lst[0], activation = act, kernel_regularizer= ln_reg)(flatten)
    drop_a = keras.layers.Dropout(dropout_coefficient)(dense_a)
    dense_b = keras.layers.Dense(dense_sizes_lst[1], activation = act, kernel_regularizer= ln_reg)(drop_a)
    drop_b = keras.layers.Dropout(dropout_coefficient)(dense_b)
    dense_c = keras.layers.Dense(dense_sizes_lst[2], activation = act, kernel_regularizer= ln_reg)(drop_b)
    drop_c = keras.layers.Dropout(dropout_coefficient)(dense_c)
    if n_layers ==4:
        dense_d = keras.layers.Dense(dense_sizes_lst[3], activation = act, kernel_regularizer= ln_reg)(drop_c)
        drop_d = keras.layers.Dropout(dropout_coefficient)(dense_d)
        # Final layer if using four dense layers
        y = keras.layers.Dense(1, activation = 'sigmoid')(drop_d)
    elif n_layers==3:
        # Final Layer if using three dense layers 
        y = keras.layers.Dense(1, activation = 'sigmoid')(drop_c)
    model = keras.models.Model(inputs=inputs, outputs=y)
    return model
