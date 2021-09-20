import tensorflow as tf
from tensor2tensor.layers import common_attention

# Source code taken from the authors of the original paper.
# Please refer to: https://github.com/drugilsberg/paccmann
def sinusoidal_positional_encoding(
    sequence_length, embed_size, name=None
):
    """
    Sinusoidal positional encoding.

    Args:
        - sequence_length: length of the sequence.
        - embed_size: size of the embedding.
        - name: optional name.
    Returns:
        A positional encoding of size `[1, sequence_length, embed_size]`.
    """
    with tf.compat.v1.variable_scope(
        name, default_name='sinusoidal_positional_encoding'
    ):
        return common_attention.get_timing_signal_1d(
            sequence_length, embed_size
        )