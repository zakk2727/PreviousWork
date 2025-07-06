
import tensorflow as tf

class Sampling(tf.keras.layers.Layer):
    def __init__(self, seed=27, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = tf.keras.random.SeedGenerator(seed)

    """
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = tf.keras.ops.shape(z_log_var)
        epsilon = tf.keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + tf.keras.ops.exp(0.5 * z_log_var) * epsilon
        
    """
    
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_log_var)[0]
        dim = tf.shape(z_log_var)[1]
        epsilon = tf.keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    