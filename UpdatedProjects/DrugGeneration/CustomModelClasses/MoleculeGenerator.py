
import tensorflow as tf
from CustomModelClasses.Sampling import Sampling

class MoleculeGenerator(tf.keras.Model):
    def __init__(self, encoder, decoder, sampling_layer, max_len, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.property_prediction_layer = tf.keras.layers.Dense(1)
        self.max_len = max_len
        #self.sampling_layer = Sampling()
        self.sampling_layer = sampling_layer

        self.train_total_loss_tracker = tf.keras.metrics.Mean(name="train_total_loss")
        self.val_total_loss_tracker = tf.keras.metrics.Mean(name="val_total_loss")

    @tf.function
    def train_step(self, data):
        #adjacency_tensor, feature_tensor, qed_tensor = data[0]
        adjacency_tensor, feature_tensor, qed_tensor = data
        graph_real = [adjacency_tensor, feature_tensor]
        self.batch_size = tf.keras.ops.shape(qed_tensor)[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, qed_pred, gen_adjacency, gen_features = self(graph_real, training=True)
            graph_generated = [gen_adjacency, gen_features]
            total_loss = self._compute_loss(z_log_var, z_mean, qed_tensor, qed_pred, graph_real, graph_generated)
            #scaled_loss = self.optimizer.scale_loss(total_loss)

        #scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
        #gradients = self.optimizer.apply(scaled_gradients, self.trainable_weights)
        #gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.train_total_loss_tracker.update_state(total_loss)
        return {"loss": self.train_total_loss_tracker.result()}

    def _compute_loss(self, z_log_var, z_mean, qed_true, qed_pred, graph_real, graph_generated):

        adjacency_real, features_real = graph_real
        adjacency_gen, features_gen = graph_generated
        #qed_true = tf.expand_dims(qed_true, -1)

        adjacency_loss = tf.keras.ops.mean(tf.keras.ops.sum(tf.keras.losses.categorical_crossentropy(adjacency_real, adjacency_gen),axis=(1, 2),))
        features_loss = tf.keras.ops.mean(tf.keras.ops.sum(tf.keras.losses.categorical_crossentropy(features_real, features_gen),axis=(1),))
        kl_loss = -0.5 * tf.keras.ops.sum(1 + z_log_var - z_mean**2 - tf.keras.ops.minimum(tf.keras.ops.exp(z_log_var), 1e6), 1)
        #kl_loss = -0.5 * tf.keras.ops.sum(1 + z_log_var - tf.keras.ops.minimum(tf.keras.ops.minimum(tf.keras.ops.exp(z_log_var), 1e6), 1))
        kl_loss = tf.keras.ops.mean(kl_loss)
        
        property_loss = tf.keras.ops.mean(tf.keras.losses.binary_crossentropy(qed_true, tf.keras.ops.squeeze(qed_pred, axis=1)))
        #property_loss = tf.keras.ops.mean(tf.keras.losses.binary_crossentropy(qed_true, qed_pred))

        graph_loss = self._gradient_penalty(graph_real, graph_generated)

        return kl_loss + property_loss + graph_loss + adjacency_loss + features_loss

    def _gradient_penalty(self, graph_real, graph_generated):
        # Unpack graphs
        adjacency_real, features_real = graph_real
        adjacency_generated, features_generated = graph_generated

        # Generate interpolated graphs (adjacency_interp and features_interp)
        alpha = tf.keras.random.uniform([self.batch_size])
        alpha = tf.keras.ops.reshape(alpha, (self.batch_size, 1, 1, 1))
        adjacency_interp = (adjacency_real * alpha) + (1 - alpha) * adjacency_generated
        alpha = tf.keras.ops.reshape(alpha, (self.batch_size, 1, 1))
        features_interp = (features_real * alpha) + (1 - alpha) * features_generated

        # Compute the logits of interpolated graphs
        with tf.GradientTape() as tape:
            tape.watch(adjacency_interp)
            tape.watch(features_interp)
            _, _, logits, _, _ = self([adjacency_interp, features_interp], training=True)

        # Compute the gradients with respect to the interpolated graphs
        grads = tape.gradient(logits, [adjacency_interp, features_interp])
        # Compute the gradient penalty
        grads_adjacency_penalty = (1 - tf.keras.ops.norm(grads[0], axis=1)) ** 2
        grads_features_penalty = (1 - tf.keras.ops.norm(grads[1], axis=2)) ** 2
        return tf.keras.ops.mean(tf.keras.ops.mean(grads_adjacency_penalty, axis=(-2, -1)) + tf.keras.ops.mean(grads_features_penalty, axis=(-1)))

    def call(self, inputs):
        z_mean, log_var = self.encoder(inputs)
        #z = Sampling()([z_mean, log_var])
        z = self.sampling_layer([z_mean, log_var])

        gen_adjacency, gen_features = self.decoder(z)

        property_pred = self.property_prediction_layer(z_mean)

        return z_mean, log_var, property_pred, gen_adjacency, gen_features