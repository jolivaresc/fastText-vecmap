'''
https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough
'''



import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from sklearn.datasets import load_wine


# Eager execution
tf.enable_eager_execution()
# Random seed
tf.set_random_seed(42)


print("TensorFlow", tf.VERSION)
print("Eager execution:",tf.executing_eagerly())


class Autoencoder(tf.keras.Model):
    def __init__(self,dim_out = 1):
        super(Autoencoder, self).__init__()
        self.h1 = tf.layers.Dense(units=20-5, activation=tf.nn.relu,use_bias=True)
        self.h2 = tf.layers.Dense(units=10-5, activation=tf.nn.relu,use_bias=True)
        self._output = tf.layers.Dense(units=dim_out)
        #self._ouput = tf.layers.Dense(units=1, activation=tf.nn.sigmoid)
    
    

    def predict(self, inputs):
        outputs = self.h1(inputs)
        outputs = self.h2(outputs)
        outputs = self._output(outputs)
        return outputs

    def loss_fn(self, inputs, targets):
        outputs = self.predict(inputs)
        loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs)
        return loss

    def grads_fn(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(inputs, targets)
        return tape.gradient(loss, self.variables)

    def fit(self, inputs, targets,hparams):
        #import matplotlib.pyplot as plt
        track_accuracy = []
        epoch_accuracy = tfe.metrics.Accuracy()
        track_loss = []

        for i in range(hparams['num_epochs']):
            grads = self.grads_fn(inputs, targets)
            hparams['optimiser'].apply_gradients(zip(grads, self.variables))
            epoch_accuracy(self.predict(inputs), targets)
            track_accuracy.append(epoch_accuracy.result())
            track_loss.append(self.loss_fn(inputs,targets))
            #epoch_accuracy.init_variables()
            if (i == 0) | ((i + 1) % hparams['verbose'] == 0):
                print('Loss at epoch %d: %f - %f' %
                      (i + 1, self.loss_fn(inputs, targets),epoch_accuracy.result()))
        '''plt.plot(track_accuracy)
        plt.grid()
        plt.show()'''

    
    
        
    
    
    
    
        
hparams = {
    'optimiser': tf.train.RMSPropOptimizer(1e-3),
    'num_epochs': 600,
    'verbose': 50
}


wines = load_wine()
# Normalize data (zero mean - 1 stddev)
wines.data = wines.data - np.mean(wines.data, axis=0)
wines.data = wines.data / np.std(wines.data, axis=0)

X = tf.constant(wines.data,dtype=np.float64)
y = tf.constant(wines.target, dtype=np.float64)

model = Autoencoder(dim_out=1)

mo

model.summary()
#model.fit(X, tf.reshape(y, [-1, 1]), hparams)


