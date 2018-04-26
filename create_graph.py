"""Genera el grafo computacional de un modelo existente
para visualizarlo en TensorBoard

"""
import tensorflow as tf

__author__ = "Olivares Castillo José Luis"

tf.reset_default_graph()
sess = tf.Session()

path = "models/es-na/2/"
saver = tf.train.import_meta_graph(path + "model2250.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint(path))


graph = tf.get_default_graph()

writer = tf.summary.FileWriter(logdir='tlogs', graph=graph)
writer.flush()
