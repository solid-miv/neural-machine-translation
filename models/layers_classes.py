"""
This file contains classes that replace some of the TensorFlow functions in the MultiHeadAttention layer,
since you cannot pass a tensor directly to the TensorFlow functions.
"""

import tensorflow as tf


# used instead of tf.shape(x)[1] in the MultiHeadAttention layer
class ShapeLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.shape(x)[1]
    
    # suppreses the warning about the mask not being used
    def compute_mask(self, inputs, mask=None):
        return None

# used instead of tf.range(x) in the MultiHeadAttention layer
class RangeLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.range(x)
    
# used instead of tf.math.not_equal(x, 0) in the MultiHeadAttention layer
class NotEqualLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.not_equal(x, 0)[:, tf.newaxis]
    
# used instead of tf.expand_dims(x, axis=1) in the MultiHeadAttention layer
class ExpandDimsLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.expand_dims(x, axis=1)

# used instead of tf.ones((x, x), tf.bool) in the MultiHeadAttention layer
class OnesLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.ones((x, x), tf.bool)

# used instead of tf.linalg.band_part(x, -1, 0) in the MultiHeadAttention layer
class BandPartLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.linalg.band_part(x, -1, 0)