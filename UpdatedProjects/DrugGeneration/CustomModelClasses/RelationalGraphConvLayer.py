
import tensorflow as tf

class RelationalGraphConvLayer(tf.keras.layers.Layer):
    def __init__(self, units=128, batch_size=64, num_replicas_in_sync = 4, activation="relu", use_bias=False, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        
        self.global_batch_size = batch_size * num_replicas_in_sync
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]

        self.kernel = self.add_weight(shape=(self.global_batch_size, bond_dim, atom_dim, self.units),initializer=self.kernel_initializer,regularizer=self.kernel_regularizer,trainable=True,name="W",dtype=tf.float32,)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.global_batch_size, bond_dim, 1, self.units),initializer=self.bias_initializer,regularizer=self.bias_regularizer,trainable=True,name="b",dtype=tf.float32,)

        self.built = True
        
    def call(self, inputs, training=False):
        adjacency, features = inputs
        # Aggregate information from neighbors
        x = tf.keras.ops.matmul(adjacency, features[:, None])
        # Apply linear transformation
        x = tf.keras.ops.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
        # Reduce bond types dim
        x_reduced = tf.keras.ops.sum(x, axis=1)
        # Apply non-linear transformation
        return self.activation(x_reduced)

    """
    
    def call(self, inputs, training=False):
        adjacency, features = inputs
        # Aggregate information from neighbors
        x = tf.matmul(adjacency, features[:, None, :, :])
        # Apply linear transformation
        x = tf.matmul(x, self.kernel)
        if self.use_bias:
            x += self.bias
        # Reduce bond types dim
        x_reduced = tf.reduce_sum(x, axis=1)
        # Apply non-linear transformation
        x_act = self.activation(x_reduced)
        return x_act
        
    """