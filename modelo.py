'''
https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough
https://github.com/keras-team/keras/blob/master/keras/engine/network.py
https://github.com/titu1994/tf-eager-examples/blob/master/notebooks/10_02_custom_layers.ipynb
'''


import tensorflow as tf
import tensorflow.contrib.eager as tfe
# import numpy as np
# from sklearn.datasets import load_wine


class Autoencoder(tf.keras.Model):
    # TODO:
    # build method

    def __init__(self, dims=None, dim_encoded=None, dropout=0.5):
        super(Autoencoder, self).__init__()
        if dims is None:
            raise ValueError(
                'Missing inputs/output dimension' +
                'Received: ' + str(dims))
        if dim_encoded is None:
            raise ValueError(
                'Missing encoded dimension' +
                'Received: ' + str(dim_encoded))

        self.dims = dims
        self.dim_encoded = dim_encoded
        self.RANDOM_SEED = 42
        self._dropout = dropout

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=self.dim_encoded,
                                      # tensorflow infiere la dimensión de los
                                      # vectores de entrada
                                      # input_shape=(self.dims,),
                                      use_bias=True,
                                      # bias_initializer=tf.zeros_initializer(),
                                      kernel_initializer=tf.truncated_normal_initializer(
                                          stddev=0.1,
                                          seed=self.RANDOM_SEED),
                                      activation=tf.nn.elu),
                tf.keras.layers.Dropout(self._dropout, seed=self.RANDOM_SEED)
            ],
            name="encoder"
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=self.dims,
                                      use_bias=True,
                                      kernel_initializer=tf.truncated_normal_initializer(
                                          stddev=0.1,
                                          seed=self.RANDOM_SEED),
                                      activation=tf.nn.tanh)
            ],
            name="decoder"
        )

    def predict(self, inputs):
        outputs = self.encoder(inputs)
        outputs = self.decoder(outputs)

        return outputs

    def loss_fn(self, inputs, targets):
        outputs = self.predict(inputs)
        loss = tf.losses.mean_squared_error(
            labels=targets, predictions=outputs)
        return loss

    def grads_fn(self, inputs, targets):
        # TODO:
        # - return loss,tape.gradients
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, targets)
        return loss, tape.gradient(loss, self.variables)

    def fit(self, inputs, targets, hparams):
        # TODO:
        # - global_step = tf.train.get_or_create_global_step()
        # - track accuracy

        if hparams["plot"]:
            import matplotlib.pyplot as plt
        track_accuracy = []
        track_loss = []
        epoch_accuracy = tfe.metrics.Accuracy()

        for i in range(hparams['num_epochs']):
            loss_val, grads = self.grads_fn(inputs, targets)
            hparams['optimizer'].apply_gradients(zip(grads, self.variables))
            epoch_accuracy(self.predict(inputs), targets)
            track_accuracy.append(epoch_accuracy.result())
            track_loss.append(loss_val)
            # epoch_accuracy.init_variables()
            if (i == 0) | ((i + 1) % hparams['verbose'] == 0):
                print('Loss at epoch %d: %f - %f' %
                      (i + 1, self.loss_fn(inputs, targets), epoch_accuracy.result()))
        if hparams["plot"]:
            plt.plot(track_loss)
            plt.title("Loss")
            plt.grid()
            plt.show()

        # Se indica que el modelo ya está
        # construido; se puede invocar el método summary()
        self.built = True
