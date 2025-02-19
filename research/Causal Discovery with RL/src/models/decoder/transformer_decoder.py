import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow_probability import distributions as distr

from ..encoder import TransformerEncoder  # Assuming this is defined elsewhere


def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    # Linear projections
    Q = Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
    K = Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
    V = Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
    
    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # [batch_size, seq_length, n_hidden/num_heads]
    
    # Multiplication
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]
    
    # Scale
    outputs = outputs / (K_.shape[-1] ** 0.5)
    
    # Activation
    outputs = tf.nn.softmax(outputs)  # num_heads*[batch_size, seq_length, seq_length]
    
    # Dropouts
    outputs = Dropout(dropout_rate)(outputs, training=is_training)
    
    # Weighted sum
    outputs = tf.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]
    
    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [batch_size, seq_length, n_hidden]
    
    # Residual connection
    outputs += inputs  # [batch_size, seq_length, n_hidden]
    
    # Normalize
    outputs = LayerNormalization(axis=2, epsilon=1e-6)(outputs)  # [batch_size, seq_length, n_hidden]
    
    return outputs


def feedforward(inputs, num_units=[2048, 512], is_training=True):
    # Inner layer
    outputs = Dense(num_units[0], activation=tf.nn.relu)(inputs)
    
    # Readout layer
    outputs = Dense(num_units[1], activation=tf.nn.relu)(outputs)
    
    # Residual connection
    outputs += inputs
    
    # Normalize
    outputs = LayerNormalization(axis=2, epsilon=1e-6)(outputs)
    
    return outputs


class TransformerDecoder:
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_dimension = config.hidden_dim
        self.input_embed = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.max_length = config.max_length

        self.initializer = tf.initializers.GlorotUniform()  # Xavier initializer

        self.is_training = is_train
        self.samples = []
        self.mask_scores = []
        self.entropy = []

    def decode(self, inputs):
        all_user_embedding = tf.reduce_mean(inputs, 1)
        inputs_with_all_user_embedding = tf.concat([inputs,
                                                   tf.tile(tf.expand_dims(all_user_embedding, 1), [1, self.max_length, 1])], -1)

        with tf.variable_scope("embedding_MCS", reuse=tf.AUTO_REUSE):
            # Embed input sequence
            W_embed = tf.get_variable("weights", [1, self.input_embed, self.input_embed], initializer=self.initializer)
            self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
            # Batch Normalization
            self.enc = LayerNormalization(axis=2)(self.embedded_input)
        
        with tf.variable_scope("stack_MCS", reuse=tf.AUTO_REUSE):
            # Blocks
            for i in range(self.num_stacks):
                with tf.variable_scope(f"block_{i}"):
                    # Multihead Attention
                    self.enc = multihead_attention(self.enc, num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)
                    
                    # Feed Forward
                    self.enc = feedforward(self.enc, num_units=[self.input_embed, self.input_embed], is_training=self.is_training)

            # Readout layer
            self.adj_prob = Dense(self.max_length)(self.enc)

            for i in range(self.max_length):
                position = tf.ones([inputs.shape[0]]) * i
                position = tf.cast(position, tf.int32)

                # Update mask
                self.mask = tf.one_hot(position, self.max_length)
                masked_score = self.adj_prob[:, i, :] - 100000000. * self.mask

                prob = distr.Bernoulli(logits=masked_score)
                sampled_arr = prob.sample()  # Batch_size, seq_length for just one node

                self.samples.append(sampled_arr)
                self.mask_scores.append(masked_score)
                self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy


if __name__ == '__main__':
    config, _ = get_config()  # Assuming this method is defined elsewhere
    print('check:', config.batch_size, config.max_length, config.input_dimension)
    input_ = tf.placeholder(tf.float32, [config.batch_size, config.max_length, config.input_dimension], name="input_channel")

    Encoder = TransformerEncoder(config, True)
    encoder_output = Encoder.encode(input_)

    # Ptr-net returns permutations (self.positions), with their log-probability for backprop
    ptr = TransformerDecoder(config, True)
    samples, logits_for_rewards = ptr.decode(encoder_output)

    graphs_gen = tf.stack(samples)
    graphs_gen = tf.transpose(graphs_gen, [1, 0, 2])
    graphs_gen = tf.cast(graphs_gen, tf.float32)
    logits_for_rewards = tf.stack(logits_for_rewards)
    logits_for_rewards = tf.transpose(logits_for_rewards, [1, 0, 2])
    log_probss = tf.nn.sigmoid_cross_entropy_with_logits(labels=graphs_gen, logits=logits_for_rewards)
    reward_probs = tf.reduce_sum(log_probss, axis=[1, 2])

    rewards = tf.reduce_sum(tf.abs(graphs_gen), axis=[1, 2])

    # Running the model
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        solver = []
        training_set = DataGenerator(solver, True)

        nb_epoch = 2

        for i in range(nb_epoch):
            input_batch = training_set.train_batch(config.batch_size, config.max_length, config.input_dimension)
            print(sess.run([tf.shape(graphs_gen), tf.shape(reward_probs)], feed_dict={input_: input_batch}))
