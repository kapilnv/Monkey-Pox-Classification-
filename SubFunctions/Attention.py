import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class CrossAttention(layers.Layer):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x1, x2 = inputs

        d = tf.cast(tf.shape(x1)[-1], tf.float32)

        # x1 -> x2
        scores_x1_to_x2 = tf.matmul(x1, x2, transpose_b=True) / tf.sqrt(d)
        attn_x1_to_x2 = tf.nn.softmax(scores_x1_to_x2, axis=-1)
        attn_x1_to_x2 = self.dropout(attn_x1_to_x2, training=training)
        output_x1_to_x2 = tf.matmul(attn_x1_to_x2, x2)

        # x2 -> x1
        scores_x2_to_x1 = tf.matmul(x2, x1, transpose_b=True) / tf.sqrt(d)
        attn_x2_to_x1 = tf.nn.softmax(scores_x2_to_x1, axis=-1)
        attn_x2_to_x1 = self.dropout(attn_x2_to_x1, training=training)
        output_x2_to_x1 = tf.matmul(attn_x2_to_x1, x1)

        return output_x1_to_x2 + output_x2_to_x1
