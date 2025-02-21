import tensorflow as tf

# Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
# Attention size = n_hidden should be a multiple of num_head
# Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]

def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
    with tf.name_scope("multihead_attention"):
        
        # Linear projections
        Q = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
        K = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
        V = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)(inputs)  # [batch_size, seq_length, n_hidden]
        
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
        outputs = tf.keras.layers.Dropout(rate=dropout_rate)(outputs)
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [batch_size, seq_length, n_hidden]
              
        # Residual connection
        outputs += inputs  # [batch_size, seq_length, n_hidden]
              
        # Normalize
        outputs = tf.keras.layers.BatchNormalization(axis=2, training=is_training, name='ln')(outputs)  # [batch_size, seq_length, n_hidden]
 
    return outputs


# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs

def feedforward(inputs, num_units=[2048, 512], is_training=True):
    with tf.name_scope("ffn"):
        # Inner layer
        outputs = tf.keras.layers.Conv1D(filters=num_units[0], kernel_size=1, activation=tf.nn.relu, use_bias=True)(inputs)
        
        # Readout layer
        outputs = tf.keras.layers.Conv1D(filters=num_units[1], kernel_size=1, activation=None, use_bias=True)(outputs)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = tf.keras.layers.BatchNormalization(axis=2, training=is_training, name='ln')(outputs)  # [batch_size, seq_length, n_hidden]
    
    return outputs


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, config, is_train):
        super(TransformerEncoder, self).__init__()
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of cities)
        self.input_dimension = config.input_dimension  # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.input_embed = config.hidden_dim  # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
        self.initializer = tf.keras.initializers.GlorotUniform()  # variables initializer
        self.is_training = is_train  # not config.inference_mode
        self.W_embed = None  # will initialize this in `build()` method

    def build(self, input_shape):
        # Initialize weights in the `build` method
        self.W_embed = self.add_weight(
            name="weights", 
            shape=[1, self.input_dimension, self.input_embed], 
            initializer=self.initializer
        )
        super(TransformerEncoder, self).build(input_shape)

    def call(self, inputs):
        with tf.name_scope("embedding"):
            # Embed input sequence using the weights initialized in `build()`
            embedded_input = tf.nn.conv1d(inputs, self.W_embed, 1, "VALID", name="embedded_input")
            
            # Batch Normalization
            enc = tf.keras.layers.BatchNormalization(axis=2, training=self.is_training, name='layer_norm')(embedded_input)
        
        with tf.name_scope("stack"):
            # Blocks
            for i in range(self.num_stacks):  # num blocks
                with tf.name_scope("block_{}".format(i)):
                    # Multihead Attention
                    enc = multihead_attention(enc, num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)
                    
                    # Feed Forward
                    enc = feedforward(enc, num_units=[4*self.input_embed, self.input_embed], is_training=self.is_training)

        # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
        return enc

    def compute_output_shape(self, input_shape):
        # Compute the output shape. This is required for custom layers in Keras.
        return (input_shape[0], input_shape[1], self.input_embed)
