'''
https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough
https://github.com/keras-team/keras/blob/master/keras/engine/network.py
'''


import tensorflow as tf
import tensorflow.contrib.eager as tfe
#import numpy as np
#from sklearn.datasets import load_wine


'''# Eager execution
tf.enable_eager_execution()
# Random seed
tf.set_random_seed(42)
np.random.seed(42)


print("TensorFlow", tf.VERSION)
print("Eager execution:", tf.executing_eagerly())'''


class Autoencoder(tf.keras.Model):
    def __init__(self, dims=None, dim_encoded=None, dropout=0.5):
        if dims is None:
            raise ValueError(
                'Missing inputs/output dimension' +
                'Received: ' + str(dims))
        if dim_encoded is None:
            raise ValueError(
                'Missing encoded dimension' +
                'Received: ' + str(dim_encoded))

        super(Autoencoder, self).__init__()
        self.dims = dims
        self.dim_encoded = dim_encoded
        self.RANDOM_SEED = 42
        self.dropout = dropout

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=self.dim_encoded,
                                      # tensorflow infiere la dimensión de los
                                      # vectores de entrada
                                      # input_shape=(self.dims,),
                                      use_bias=True,
                                      kernel_initializer=tf.truncated_normal_initializer(
                                          stddev=0.1,
                                          seed=self.RANDOM_SEED
                                      ),
                                      activation=tf.nn.tanh)
            ],
            name="encoder"
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=self.dims,
                                      use_bias=True,
                                      kernel_initializer=tf.truncated_normal_initializer(
                                          stddev=0.1,
                                          seed=self.RANDOM_SEED
                                      ),
                                      activation=tf.nn.tanh)
            ],
            name="decoder"
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout,
                                               seed=self.RANDOM_SEED)

    def predict(self, inputs, training=False):
        outputs = self.encoder(inputs)
        outputs = self.dropout(inputs)
        outputs = self.decoder(outputs)

        return outputs

    def loss_fn(self, inputs, targets):
        outputs = self.predict(inputs)
        loss = tf.losses.mean_squared_error(
            labels=targets, predictions=outputs)
        return loss

    def grads_fn(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, targets)
        return tape.gradient(loss, self.variables)

    def fit(self, inputs, targets, hparams):
        if hparams["plot"]:
            import matplotlib.pyplot as plt
        track_accuracy = []
        epoch_accuracy = tfe.metrics.Accuracy()
        track_loss = []

        for i in range(hparams['num_epochs']):
            grads = self.grads_fn(inputs, targets)
            hparams['optimizer'].apply_gradients(zip(grads, self.variables))
            epoch_accuracy(self.predict(inputs), targets)
            track_accuracy.append(epoch_accuracy.result())
            track_loss.append(self.loss_fn(inputs, targets))
            #epoch_accuracy.init_variables()
            if (i == 0) | ((i + 1) % hparams['verbose'] == 0):
                print('Loss at epoch %d: %f - %f' %
                      (i + 1, self.loss_fn(inputs, targets), epoch_accuracy.result()))
        if hparams["plot"]:
            plt.plot(track_loss)
            plt.grid()
            plt.show()

        # Se indica que el modelo ya está
        # construido; se puede invocar el método summary()
        self.built = True

'''
hparams = {
    'optimizer': tf.train.RMSPropOptimizer(1e-4, centered=True),
    'num_epochs': 2000,
    'verbose': 50,
    'plot': True
}


wines = load_wine()
# Normalize data (zero mean - 1 stddev)
wines.data = wines.data - np.mean(wines.data, axis=0)
wines.data = wines.data / np.std(wines.data, axis=0)

X = tf.constant(wines.data, dtype=np.float)
y = tf.constant(wines.target, dtype=np.float)

model = Autoencoder(dims=X.shape[1], dim_encoded=2,)


model.fit(X, X, hparams)

model.summary()
'''