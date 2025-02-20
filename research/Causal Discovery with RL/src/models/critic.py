import tensorflow as tf
import numpy as np

class Critic(object):
    def __init__(self, config, is_train):
        self.config = config

        # Data config
        self.batch_size = config.batch_size 
        self.max_length = config.max_length 
        self.input_dimension = config.input_dimension 

        # Network config
        self.input_embed = config.hidden_dim 
        self.num_neurons = config.hidden_dim 
        self.initializer = tf.keras.initializers.GlorotUniform()  # Xavier initializer equivalent

        # Baseline setup
        self.init_baseline = 0.

    def predict_rewards(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = tf.reduce_mean(encoder_output, axis=1) 

        with tf.variable_scope("ffn", reuse=tf.AUTO_REUSE):
            # Feedforward layer 1
            h0 = tf.keras.layers.Dense(self.num_neurons, activation=tf.nn.relu, kernel_initializer=self.initializer)(frame)
            # Weight and bias for the output layer
            w1 = tf.Variable(self.initializer([self.num_neurons, 1]), name="w1")
            b1 = tf.Variable(self.init_baseline, name="b1")
            
            # Compute predictions
            self.predictions = tf.squeeze(tf.matmul(h0, w1) + b1)

        return self.predictions
