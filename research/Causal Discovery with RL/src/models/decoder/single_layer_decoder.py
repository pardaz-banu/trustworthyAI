import tensorflow as tf

class SingleLayerDecoder(object):

    def __init__(self, config, is_train):
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

    def decode(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        
        # Variable initialization using tf.keras
        W_l = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[self.input_embed, self.decoder_hidden_dim]), name='weights_left')
        W_r = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[self.input_embed, self.decoder_hidden_dim]), name='weights_right')
        U = tf.Variable(tf.keras.initializers.GlorotUniform()(shape=[self.decoder_hidden_dim]), name='U')  # Aggregate across decoder hidden dim

        # Compute dot products
        dot_l = tf.einsum('ijk, kl->ijl', encoder_output, W_l)
        dot_r = tf.einsum('ijk, kl->ijl', encoder_output, W_r)

        # Tiling the dot products for element-wise addition
        tiled_l = tf.tile(tf.expand_dims(dot_l, axis=2), (1, 1, self.max_length, 1))
        tiled_r = tf.tile(tf.expand_dims(dot_r, axis=1), (1, self.max_length, 1, 1))

        # Apply activation function based on user input
        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = tf.nn.tanh(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = tf.nn.relu(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = tf.einsum('ijkl, l->ijk', final_sum, U)    # Readability

        # Bias handling
        if self.bias_initial_value is None:    # Randomly initialize the learnable bias
            self.logit_bias = tf.Variable(tf.zeros([1]), name='logit_bias')
        elif self.use_bias_constant:    # Constant bias
            self.logit_bias = tf.constant([self.bias_initial_value], dtype=tf.float32, name='logit_bias')
        else:    # Learnable bias with initial value
            self.logit_bias = tf.Variable([self.bias_initial_value], dtype=tf.float32, name='logit_bias')

        if self.use_bias:    # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        # Sampling using Bernoulli distribution
        for i in range(self.max_length):
            position = tf.ones([encoder_output.shape[0]]) * i
            position = tf.cast(position, tf.int32)

            # Update mask
            self.mask = tf.one_hot(position, self.max_length)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = tf.distributions.Bernoulli(logits=masked_score)    # probs input probability, logit input log_probability

            sampled_arr = prob.sample()    # Batch_size, sequence_length for just one node

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy
