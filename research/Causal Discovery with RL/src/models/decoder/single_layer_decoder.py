import tensorflow as tf
from tensorflow.keras import layers

class SingleLayerDecoder(layers.Layer):
    def __init__(self, config, is_train, **kwargs):
        super(SingleLayerDecoder, self).__init__(**kwargs)
        
        # Initialize configuration variables
        self.batch_size = config.batch_size    # batch size
        self.max_length = config.max_length    # input sequence length (number of cities)
        self.input_dimension = config.hidden_dim
        self.input_embed = config.hidden_dim    # dimension of embedding space (actor)
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.decoder_activation = config.decoder_activation
        self.use_bias = config.use_bias
        self.bias_initial_value = config.bias_initial_value
        self.use_bias_constant = config.use_bias_constant

        self.is_training = is_train

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

        # Weights are now initialized inside the `build()` method (Keras' recommended way)
        self.W_l = None
        self.W_r = None
        self.U = None
        self.logit_bias = None

    def build(self, input_shape):
        # Initialize weights here using the Keras API
        self.W_l = self.add_weight(
            name='weights_left', shape=[self.input_embed, self.decoder_hidden_dim], 
            initializer=tf.keras.initializers.GlorotUniform())
        self.W_r = self.add_weight(
            name='weights_right', shape=[self.input_embed, self.decoder_hidden_dim], 
            initializer=tf.keras.initializers.GlorotUniform())
        self.U = self.add_weight(
            name='U', shape=[self.decoder_hidden_dim], 
            initializer=tf.keras.initializers.GlorotUniform())
        
        # Bias handling
        if self.bias_initial_value is None:
            self.logit_bias = self.add_weight(
                name='logit_bias', shape=[1], initializer=tf.zeros_initializer())
        elif self.use_bias_constant:
            self.logit_bias = self.add_weight(
                name='logit_bias', shape=[1], initializer=tf.constant_initializer(self.bias_initial_value))
        else:
            self.logit_bias = self.add_weight(
                name='logit_bias', shape=[1], initializer=tf.keras.initializers.GlorotUniform())

    def call(self, encoder_output):
        # Compute dot products
        dot_l = tf.einsum('ijk, kl->ijl', encoder_output, self.W_l)
        dot_r = tf.einsum('ijk, kl->ijl', encoder_output, self.W_r)

        # Tiling the dot products for element-wise addition
        tiled_l = tf.tile(tf.expand_dims(dot_l, axis=2), (1, 1, self.max_length, 1))
        tiled_r = tf.tile(tf.expand_dims(dot_r, axis=1), (1, self.max_length, 1, 1))

        # Apply activation function based on user input
        if self.decoder_activation == 'tanh':
            final_sum = tf.nn.tanh(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = tf.nn.relu(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = tf.einsum('ijkl, l->ijk', final_sum, self.U)

        # Bias handling
        if self.use_bias:
            logits += self.logit_bias

        self.adj_prob = logits

        # Sampling using Bernoulli distribution
        for i in range(self.max_length):
            position = tf.ones([encoder_output.shape[0]]) * i
            position = tf.cast(position, tf.int32)

            # Update mask
            self.mask = tf.one_hot(position, self.max_length)

            masked_score = self.adj_prob[:, i, :] - 100000000. * self.mask
            prob = tf.distributions.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, sequence_length for just one node

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy
