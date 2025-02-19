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
        self.embedding_layer = Dense(self.input_embed, activation=tf.nn.relu, kernel_initializer=self.initializer)
        self.attention_blocks = [multihead_attention for _ in range(self.num_stacks)]
        self.ffn_blocks = [feedforward for _ in range(self.num_stacks)]
        
    def call(self, inputs, training=True):
        # Apply embedding and attention/ffn blocks
        all_user_embedding = tf.reduce_mean(inputs, 1)
        inputs_with_all_user_embedding = tf.concat([inputs, 
                                                   tf.tile(tf.expand_dims(all_user_embedding, 1), [1, self.max_length, 1])], -1)

        embedded_input = self.embedding_layer(inputs_with_all_user_embedding)
        enc = embedded_input
        
        # Apply multi-head attention and feedforward blocks
        for i in range(self.num_stacks):
            enc = self.attention_blocks[i](enc, num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)
            enc = self.ffn_blocks[i](enc, num_units=[self.input_embed, self.input_embed], is_training=self.is_training)
        
        # Readout layer
        adj_prob = Dense(self.max_length, activation=None)(enc)

        samples = []
        mask_scores = []
        entropy = []

        for i in range(self.max_length):
            position = tf.ones([inputs.shape[0]]) * i
            position = tf.cast(position, tf.int32)
            mask = tf.one_hot(position, self.max_length)

            masked_score = adj_prob[:, i, :] - 1e8 * mask
            prob = distr.Bernoulli(logits=masked_score)
            sampled_arr = prob.sample()

            samples.append(sampled_arr)
            mask_scores.append(masked_score)
            entropy.append(prob.entropy())

        return samples, mask_scores, entropy


# Main execution to test the decoder
if __name__ == '__main__':
    config = {
        'batch_size': 32,
        'max_length': 10,
        'hidden_dim': 64,
        'num_heads': 8,
        'num_stacks': 4
    }

    input_ = tf.random.normal([config['batch_size'], config['max_length'], config['hidden_dim']])
    decoder = TransformerDecoder(config, is_train=True)
    samples, mask_scores, entropy = decoder(input_)

    # Print the outputs (check shapes of the returned tensors)
    print("Samples:", [sample.shape for sample in samples])
    print("Mask Scores:", [score.shape for score in mask_scores])
    print("Entropy:", [e.shape for e in entropy])
