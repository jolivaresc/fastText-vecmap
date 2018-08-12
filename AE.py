# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import utils
from modelo import Autoencoder


__author__ = "Olivares Castillo José Luis"


# Eager execution
tf.enable_eager_execution()
# Random seed
tf.set_random_seed(42)
np.random.seed(42)


print("TensorFlow", tf.VERSION)
print("Eager execution:", tf.executing_eagerly())


# Cargar lexicón semilla
source_lex = "es-na.train"
words_src_lexicon, words_trg_lexicon = utils.get_lexicon(source_lex)
print("\nSize of lexicon:", set(words_src_lexicon).__len__())
#print(len(words_src_lexicon), len(words_trg_lexicon))


# Cargar archivo con vectores Node2Vec en español
source_str = "es.n2v"
# Cargar vectores de source_str
source_vec = utils.open_file(source_str)
words_src, source_vec = utils.read(source_vec, is_zipped=False)

# Lista de palabras en español del lexicon semilla
eval_src = words_src_lexicon
# Obtener vectores de palabras del lexicon
src_vec = utils.get_vectors(eval_src, words_src, source_vec)
print("\nSource_vec: " + source_str, "\tShape:", src_vec.shape)


# Cargar archivo con vectores Node2Vec en náhuatl
target_str = "na.n2v"
# Cargar vectores de target_str
target_vec = utils.open_file(source_str)
words_trg, target_vec = utils.read(target_vec, is_zipped=False)

# lista de palabras en náhuatl del lexicon semilla
eval_target = words_trg_lexicon
# Obtener vectores de palabras del lexicon
trg_vec = utils.get_vectors(eval_target, words_trg, target_vec)
print("Target_vec: " + target_str, "\tShape:", trg_vec.shape, "\n")


##########################

# Se instancia un nuevo objeto (Autoencoder)
ae = Autoencoder(dims=src_vec.shape[1],  # Dimensión de entrada/salida
                 dim_encoded=350,       # Dimensión de vector latente
                 dropout=0.51)

# Hiper-parámetros del modelo
hparams = {
    'optimizer': tf.train.RMSPropOptimizer(1e-4, centered=True),
    'num_epochs': 600,
    'verbose': 50,
    'plot': False
}

# Se castean los vectores
# de entrenamiento (numpy.array) a tensores (tf.Tensor)
X = tf.constant(src_vec, dtype=tf.float32)
y = tf.constant(trg_vec, dtype=tf.float32)

# TODO: compilar modelo
# ae.compile(optimizer=tf.train.RMSPropOptimizer(1e-4, centered=True),
#            loss=tf.losses.mean_squared_error,
#            metrics=["accuracy"])

# Se entrena el modelo
ae.fit(X, y, hparams)


# Guardar parámetros aprendidos de la red
ae.save_weights("ae-weights.h5", save_format='h5', overwrite=True)
