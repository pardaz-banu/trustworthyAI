import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras import layers
from tensorflow import distributions as distr

# Assuming TransformerEncoder is a Keras-based implementation or similar.

def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    # Multihead Attention implementation using TensorFlow 2.x API
    batch_size, seq_length, _ = inputs.shape
    
    # Linear projections
    Q = Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
    K = Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
    V = Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
    
    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [num_heads * batch_size, seq_length, n_hidden/num_heads]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    # Scaled dot-product attention
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [num_heads * batch_size, seq_length, seq_length]
    outputs = outputs / tf.sqrt(tf.cast(K_.shape[-1], tf.float32))  # Scale

    # Softmax to get attention scores
    outputs = tf.nn.softmax(outputs)

    # Dropout
    outputs = Dropout(dropout_rate)(outputs, training=is_training)
    
    # Weighted sum of values
    outputs = tf.matmul(outputs, V_)  # [num_heads * batch_size, seq_length, n_hidden/num_heads]
    
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [batch_size, seq_length, n_hidden]
    
    # Residual connection
    outputs = outputs + inputs  # [batch_size, seq_length, n_hidden]

    # Normalize
    outputs = LayerNormalization(axis=-1)(outputs)

    return outputs


def feedforward(inputs, num_units=[2048, 512], is_training=True):
    # Feedforward network using Keras layers
    x = Dense(num_units[0], activation=tf.nn.relu)(inputs)
    x = Dense(num_units[1], activation=tf.nn.relu)(x)
    x = x + inputs  # Residual connection
    x = LayerNormalization(axis=-1)(x)  # Normalize
    return x


class TransformerDecoder(tf.keras.Model):
    def __init__(self, config, is_train):
        super(TransformerDecoder, self).__init__()
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_embed = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.is_training = is_train

        # Initialize embedding layer and attention/ffn stacks
        self.embedding_layer = Dense(self.input_embed, activ
