
# coding: utf-8


import tensorflow as tf
import numpy as np
import utils
import time
from collections import Counter


start_time = time.time()


__author__ = "Olivares Castillo José Luis"

#tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

print("TensorFlow version: {}".format(tf.VERSION))
#print("Eager execution: {}".format(tf.executing_eagerly()))

if tf.test.gpu_device_name():
    print("GPU disponible")


#words_scr_lexicon, words_trg_lexicon = utils.get_lexicon("es-na.test")
words_scr_lexicon, words_trg_lexicon = utils.get_lexicon("en-it.test")
#print(len(words_scr_lexicon), len(words_trg_lexicon))


#source_str = "es.n2v"
source_str = "en.norm.fst"
source_vec = utils.open_file(source_str)
words_src, source_vec = utils.read(source_vec, is_zipped=False)
eval_src = list(set(words_scr_lexicon))
src_vec = utils.get_vectors(eval_src, words_src, source_vec)
print("source_vec: " + source_str)
#print(src_vec.shape)


#target_str = "na.n2v"
target_str = "it.norm.fst"
target_vec = utils.open_file(target_str)
words_trg, target_vec = utils.read(target_vec, is_zipped=False)
print("target_vec: " + target_str)
#eval_it = list(set(it))
#trg_vec = get_vectors(eval_it, words_it, it_vec)
#print(target_vec.shape)


test_vectors = src_vec


tf.reset_default_graph()
sess = tf.Session()
#path="models/en-it/3/"
#path = "models/es-na/model_joyce/"
saver = tf.train.import_meta_graph(path + "model2250.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint(path))


graph = tf.get_default_graph()
X = graph.get_tensor_by_name("input/input_es:0")
kprob = graph.get_tensor_by_name("dropout_prob:0")


#print([n.name for n in graph.as_graph_def().node])


output_NN = graph.get_tensor_by_name("nah_predicted/BiasAdd:0")
#output_NN = graph.get_tensor_by_name("xw_plus_b_1:0")
#output_NN = graph.get_tensor_by_name("nah_predicted:0")
#code = graph.get_tensor_by_name("xw_plus_b_2:0")
#print(output_NN)

feed_dict = {X: test_vectors, kprob: 1}
pred = sess.run(output_NN, feed_dict)
#print(pred.shape)


top_10 = [utils.get_top10_vectors(pred[_], target_vec)
          for _ in range(pred.shape[0])]


#get_ipython().run_cell_magic('time', '', 'closest = [utils.closest_word_to(top_10[_], words_trg) for _ in range(pred.shape[0])]')
closest = [utils.closest_word_to(top_10[_], words_trg)
           for _ in range(pred.shape[0])]


resultados = {palabra_en: top_10_it for (
    palabra_en, top_10_it) in zip(eval_src, closest)}


gold = utils.gold_dict(words_scr_lexicon, words_trg_lexicon)


p1, p5, p10 = 0, 0, 0
list_en_eval = list(resultados.keys())
hits, not_found = [], []

for palabra_gold in list_en_eval:
    for i in gold[palabra_gold]:
        if i in resultados[palabra_gold]:
            hits.append(resultados[palabra_gold].index(i))
    if hits.__len__() > 0:
        if min(hits) == 0:
            p1 += 1
            p5 += 1
            p10 += 1
        if min(hits) >= 1 and min(hits) <= 5:
            p5 += 1
            p10 += 1
        if min(hits) > 5 and min(hits) < 10:
            p10 += 1
    else:
        not_found.append(palabra_gold)
    hits.clear()

length = list_en_eval.__len__()
print("not found:", not_found.__len__(),
      "-", not_found.__len__() / length, "%")
print("P@1:", p1, "\tP@5:", p5, "\tP@10:", p10)
print("P@1:", p1 / length, "\tP@5:", p5 / length, "\tP@10:", p10 / length)
e = time.time() - start_time
print("Time: %02d:%02d:%02d" % (e // 3600, (e % 3600 // 60), (e % 60 // 1)))
