# coding: utf-8
"""Encoder

Se calculan los vectores resultantes del encoder y se guardan en un archivo

"""

import os
from collections import Counter

import numpy as np
import tensorflow as tf

import utils


__author__ = "Olivares Castillo José Luis"

#tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

print("TensorFlow version: {}".format(tf.VERSION))
#print("Eager execution: {}".format(tf.executing_eagerly()))

if tf.test.gpu_device_name():
    print("GPU disponible")


source_str = "it.fst"
source_vec = utils.open_file(source_str)
words_src, source_vec = utils.read(source_vec, is_zipped=False)
print("source_vec: " + source_str)


tf.reset_default_graph()
sess = tf.Session()
#path="models/en-it/2/"
#path = "models/es-na/encoding_n2v/na-es/3/"
path = "models/en-it/encoder/it-en/6/"
#path = "models/es-na/encoding_n2v/na-es/"
saver = tf.train.import_meta_graph(path + "model2250.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint(path))


graph = tf.get_default_graph()
X = graph.get_tensor_by_name("input/input_es:0")
kprob = graph.get_tensor_by_name("dropout_prob:0")


#print([n.name for n in graph.as_graph_def().node])


output_NN = graph.get_tensor_by_name("h1/Elu:0")


feed_dict = {X: source_vec, kprob: 1}
pred = sess.run(output_NN, feed_dict)

print(pred.shape[0], pred.shape[1])

path = os.getcwd() + "/" + path + source_str + ".encoded"
print("writing encoded vectors in", path)


with open(path, "w") as f:
    f.write(str(pred.shape[0]) + " " + str(pred.shape[1]) + "\n")
    for i in range(pred.shape[0]):
        if i.__ne__(pred.shape[0] - 1):
            f.write(words_src[i] + " " + " ".join(map(str, pred[i])) + "\n")
        else:
            f.write(words_src[i] + " " + " ".join(map(str, pred[i])))
