
import tensorflow as tf

class MoleculeGeneratorDecoder(tf.keras.layers.Layer):
    def __init__(self, dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape, **kwargs):
        super().__init__(**kwargs)
        
        self.latent_inputs = tf.keras.Input(shape=(latent_dim,))
        
        self.dense_units = dense_units
        
        self.latent_layers = [tf.keras.Sequential([tf.keras.layers.Dense(units, activation='tanh'),tf.keras.layers.Dropout(dropout_rate)]) for units in dense_units]
        
        self.x_adjacency_dense = tf.keras.layers.Dense(tf.math.reduce_prod(adjacency_shape).numpy())
        self.x_adjacency_reshape  = tf.keras.layers.Reshape(adjacency_shape)
        self.x_adjacency_activation = tf.keras.layers.Softmax(axis=1)
        
        self.x_features_dense = tf.keras.layers.Dense(tf.math.reduce_prod(feature_shape).numpy())
        self.x_features_reshape = tf.keras.layers.Reshape(feature_shape)
        self.x_features_activation = tf.keras.layers.Softmax(axis=2)
        
    def call(self, inputs, training=False):
        self.latent_inputs = inputs
        
        x = self.latent_inputs
        for i in range(len(self.latent_layers)):
            x = self.latent_layers[i](x)
            
        x_adjacency = self.x_adjacency_dense(x)
        x_adjacency = self.x_adjacency_reshape(x_adjacency)
        #x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2
        x_adjacency = (x_adjacency + tf.keras.ops.transpose(x_adjacency, (0, 1, 3, 2))) / 2
        x_adjacency = self.x_adjacency_activation(x_adjacency)
        
        x_features = self.x_features_dense(x)
        x_features = self.x_features_reshape(x_features)
        x_features = self.x_features_activation(x_features)
        
        return x_adjacency, x_features
    
    