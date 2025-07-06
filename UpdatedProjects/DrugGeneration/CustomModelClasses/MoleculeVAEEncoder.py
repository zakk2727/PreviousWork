
import tensorflow as tf
from CustomModelClasses.RelationalGraphConvLayer import RelationalGraphConvLayer

class MoleculeGeneratorEncoder(tf.keras.layers.Layer):
    def __init__(self, gconv_units, gconv_layers, latent_dim, adjacency_shape, feature_shape, dense_units, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        
        self.adjacency = tf.keras.layers.Input(shape=adjacency_shape)
        self.features = tf.keras.layers.Input(shape=feature_shape)
        
        self.gconv_units = gconv_units
        self.dense_units = dense_units
        
        #self.gconv_layers = [RelationalGraphConvLayer() for _ in gconv_units]
        
        self.gconv_layers = gconv_layers
        
        self.global_avg_poold_layer = tf.keras.layers.GlobalAveragePooling1D()
        
        self.dense_layers = [tf.keras.Sequential([tf.keras.layers.Dense(units, activation='relu'),tf.keras.layers.Dropout(dropout_rate)]) for units in dense_units]
        
        self.z_mean = tf.keras.layers.Dense(latent_dim, dtype="float32", name="z_mean")
        self.log_var = tf.keras.layers.Dense(latent_dim, dtype="float32", name="log_var")
    
    @tf.function    
    def call(self, inputs, training=False):
        self.adjacency, self.features = inputs
        # Propagate through one or more graph convolutional layers
        features_transformed = self.features
        
        for i in range(len(self.gconv_layers)):
            features_transformed = self.gconv_layers[i]([self.adjacency, features_transformed])
            
        # Reduce 2-D representation of molecule to 1-D
        x = self.global_avg_poold_layer(features_transformed)
        
        # Propagate through one or more densely connected layers
        for o in range(len(self.dense_layers)):
            x = self.dense_layers[o](x)
            
        z_mean = self.z_mean(x)
        log_var = self.log_var(x)
        
        return z_mean, log_var