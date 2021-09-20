import tensorflow as tf
from tensorflow import keras


# Using notation stated in the paper by Zheng et al
# n: number of tokens in each SMILES (after padding). It is THE SAME for all SMILES
# d: dimensionality of each token vector computed using mol2vec. Using the pretrained model, d=300
# u: number of hidden units per LSTM nn. In a Bidirectional LSTM, the final number of hidden units would be 2u (uu)
# da: number of parameters of weight vectors W1 and w2, as stated in the paper
# r: number of heads in the multi-head attention, also called number of attention rows
class SelfAttentiveLayer(keras.layers.Layer):
    def __init__(self, r, da, uu, **kwargs):
        super(SelfAttentiveLayer, self).__init__(**kwargs)
        self.da = da
        self.r = r
        self.uu = uu

    def build(self, input_shape):
        # concat of outputs of BiLSTM, with shape (batch_size, n, uu)
        assert isinstance(input_shape, tf.TensorShape)
        n, uu = int(input_shape[1]), int(input_shape[2])
        self.W1 = self.add_weight(name='W1_attention',
                                  #shape=tf.TensorShape((self.da, vector_size)), ---->>> WRONG!!!! SHOULD BE (da, uu)
                                  shape=(self.da, self.uu),
                                  initializer='glorot_uniform', trainable=True)
        self.W2 = self.add_weight(name='W2_attention',
                                  shape=(self.r, self.da),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttentiveLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # ignore kwargs inorder to keep same signature with meta class
        # the input should have already been concatenated!
        if isinstance(inputs, list):
            raise TypeError('Argument `inputs` should have already been concatenated. '
                            'Do not directly pass the output of LSTM or GRU to this layer, '
                            'use `tf.concat` first to merge them into single tensor.'
                            'If your input sequences have variable length, please padd first.')
        # TODO: add mask for padded sequence
        H = inputs
        # input : (batch_size, n, uu)
        # 1. transpose to (batch_size, uu, n)
        H_trans = tf.transpose(inputs, perm=[0, 2, 1])
        # 2. matmul with W1: (self.da, uu) * (batch_size, uu, n) = (batch_size, self.da, n)
        # 3. tanh activation
        shape_H_trans = tf.shape(H_trans)
        new_W1 = tf.expand_dims(self.W1,0)
        #W1_transformed = tf.broadcast_to(new_W1, [shape_H_trans[0], self.da, self.uu])
        W1_transformed = tf.tile(new_W1, [shape_H_trans[0], 1,1])
        after_w1 = tf.tanh(W1_transformed @ H_trans)
        # 4. matmul with W2: (self.r, self.da) * (batch_size, self.da, n) = (batch_size, self.r, n)
        # 5. softmax activation
        # 6. the output matrix A will has a shape of (batch_size, self.r, n)
        shape_after_w1 = tf.shape(after_w1)
        new_W2 = tf.expand_dims(self.W2,0)
        #W2_transformed = tf.broadcast_to(new_W2, [shape_after_w1[0], self.r, self.da])
        W2_transformed = tf.tile(new_W2, [shape_after_w1[0], 1,1])
        A = tf.nn.softmax(W2_transformed @ after_w1, axis=1)
        # 7. matmul A with input H to get output Ma:
        # (batch_size, self.r, n) * (batch_size, n, uu) = (batch_size, self.r, uu)
        Ma = A @ H
        # NO BIAS
        return Ma
