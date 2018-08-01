# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import utils
import modelo


__author__ = "Olivares Castillo José Luis"

# Random seed
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
tf.reset_default_graph()


print("TensorFlow", tf.VERSION)
if tf.test.gpu_device_name():
  print("GPU disponible")

# cargar lexicón semilla
source_lex = "es-na.train"
words_src_lexicon, words_trg_lexicon = utils.get_lexicon(source_lex)
print("size of lexicon:", set(words_src_lexicon).__len__())
print(len(words_src_lexicon), len(words_trg_lexicon))


# Archivo con vectores Node2Vec en español
source_str = "es.n2v"
# Cargar vectores de source_str
source_vec = utils.open_file(source_str)
words_src, source_vec = utils.read(source_vec, is_zipped=False)

# lista de palabras en español del lexicon semilla
#eval_src = list(set(words_src_lexicon))
eval_src = words_src_lexicon
# Obtener vectores de palabras del lexicon
src_vec = utils.get_vectors(eval_src, words_src, source_vec)
print("\nSource_vec: " + source_str, "\tShape:", src_vec.shape)


# Archivo con vectores Node2Vec en náhuatl
target_str = "na.n2v"
# Cargar vectores de target_str
target_vec = utils.open_file(source_str)
words_trg, target_vec = utils.read(target_vec, is_zipped=False)

# lista de palabras en náhuatl del lexicon semilla
#eval_src = list(set(words_src_lexicon))
eval_target = words_trg_lexicon
# Obtener vectores de palabras del lexicon
trg_vec = utils.get_vectors(eval_target, words_trg, target_vec)
print("Target_vec: " + target_str, "\tShape:", trg_vec.shape)


################ 

# Hiper-parámetros
LEARNING_RATE = 0.001
EPOCHS = 600
# Dimensión de vectores de entrada (número de neuronas en capa de entrada).
DIM_INPUT = src_vec.shape[1]
DIM_H1 = 350
DIM_OUTPUT = trg_vec.shape[1]
DROPOUT = 0.51

# Entradas y salidas del modelo
with tf.name_scope('input'):
    X = tf.placeholder(shape=[None, DIM_INPUT],
                       dtype=tf.float64, name='input_es')
    y = tf.placeholder(shape=[None, DIM_OUTPUT],
                       dtype=tf.float64, name='target_na')


# DROPOUT
kprob = tf.placeholder(tf.float64, name='dropout_prob')


# Capas del modelo
# y = x*w+b
dense1 = tf.layers.dense(inputs=X,
                        units=DIM_H1,
                        activation=tf.nn.elu,
                        use_bias=True,
                        kernel_initializer=tf.truncated_normal_initializer(
                            stddev=0.1, seed=RANDOM_SEED),
                        name="h1_layer")
# Aplicar dropout
dense1 = tf.layers.dropout(dense1, rate=DROPOUT, seed=RANDOM_SEED)


# Capa de salida
output_layer = tf.layers.dense(inputs=dense1,
                                units=DIM_OUTPUT,
                                activation=tf.nn.tanh,
                                use_bias=True,
                                kernel_initializer=tf.truncated_normal_initializer(
                                    stddev=0.1, seed=RANDOM_SEED),
                                name="nah_predicted")


# Función de error
loss = tf.reduce_mean(tf.squared_difference(output_layer, y), name="loss")


# Optimiser
optimiser = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,centered=True)


# Compute gradients
gradients, variables = zip(*optimiser.compute_gradients(loss))

gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

# Apply processed gradients to optimizer.
train_op = optimiser.apply_gradients(zip(gradients, variables))


# Sesión tensorflow
sess = tf.Session()


# Para guardar el modelo
saver = tf.train.Saver()


# Ejecutando sesión
sess.run(tf.global_variables_initializer())


# backprop
for i in range(EPOCHS):
    _loss, _ = sess.run([loss, train_op], feed_dict={
        X:src_vec,
        y:trg_vec,
        kprob:DROPOUT
    })

    print("Epoch:", i, "/", EPOCHS, "\tLoss:", _loss)

model = "ae-weights"

SAVE_PATH = "./" + model + ".ckpt"
print("save path", SAVE_PATH)
save_model = saver.save(sess, SAVE_PATH)
