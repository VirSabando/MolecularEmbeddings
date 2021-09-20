import tensorflow as tf
from tensorflow import keras

# Custom sequence attention layer, based on the source code provided by Oskooei et al.
# Please refer to: https://github.com/drugilsberg/paccmann
# n: number of tokens in each SMILES tokenized sequence (after padding). It is THE SAME for all SMILES
# u: number of hidden units == embedding size
# da: number of parameters of weight vectors W1 and W2
class Sequence_Attention_Layer(keras.layers.Layer):
    def __init__(self, da, u, **kwargs):
        super(Sequence_Attention_Layer, self).__init__(**kwargs)
        self.da = da
        self.u = u

    def build(self, input_shape):
        assert isinstance(input_shape, tf.TensorShape)
        n, u = int(input_shape[1]), int(input_shape[2])
        self.W = self.add_weight(name='W_attention',
                                  shape=(self.u, self.da),
                                  initializer='glorot_normal', trainable=True)
        self.B = self.add_weight(name='B_attention',
                                  shape=(1,self.da),
                                  initializer='glorot_normal',
                                  trainable=True)
        self.U = self.add_weight(name='U_attention',
                                  shape=(self.da,1),
                                  initializer='glorot_normal',
                                  trainable=True)
        super(Sequence_Attention_Layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            raise TypeError('Argument `inputs` should have already been concatenated. '
                            'Do not directly pass the output of LSTM or GRU to this layer, '
                            'use `tf.concat` first to merge them into single tensor.'
                            'If your input sequences have variable length, please padd first.')
        H = inputs
        # input : (batch_size, n, u)
        # 1. matmul with W: (batch_size, n, u) * (u, da) = (batch_size, n, da)
        # 2. sum B
        # 3. tanh activation fn
        shape_H = tf.shape(H)
        new_W = tf.expand_dims(self.W,0)
        new_B = tf.expand_dims(self.B,0)
        W_transformed = tf.tile(new_W, [shape_H[0],1,1])
        B_transformed = tf.tile(new_B, [shape_H[0],1,1])
        V = tf.tanh((H @ W_transformed) + B_transformed)
        # 4. matmul with U: (batch_size, n, da) * (self.da) = (batch_size, n)
        # 5. softmax activation
        # 6. the output matrix VU will has a shape of (batch_size, n)
        shape_V = tf.shape(V)
        new_U = tf.expand_dims(self.U,0)
        U_transformed = tf.tile(new_U, [shape_V[0],1,1])
        VU = tf.nn.softmax(V @ U_transformed, axis=1) # vu es alphas
        # input H: (batch_size, n, u)
        # 1. transpose to (batch_size, u, n)
        H_trans = tf.transpose(inputs, perm=[0, 2, 1])
        # 7. matmul input H and VU to get output Ma:
        # (batch_size, n, u) * (batch_size, n) = (batch_size, self.n, u)
        Ma = H * VU
        return (Ma, VU)
                