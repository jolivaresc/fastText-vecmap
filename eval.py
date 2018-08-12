
# coding: utf-8


import time
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE as TSNE_SK
from Node2Vec.tsne import tsne as TSNE
from sklearn.manifold import MDS, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from random import choice
from modelo import Autoencoder

import utils


__author__ = "Olivares Castillo José Luis"

start_time = time.time()

# Eager execution
tf.enable_eager_execution()
# Random seed
tf.set_random_seed(42)
np.random.seed(42)


print("TensorFlow", tf.VERSION)
print("Eager execution:", tf.executing_eagerly())

# Cargar lexicón de evaluación
#source_lex = "en-it.test"
source_lex = "es-na.test"
words_scr_lexicon, words_trg_lexicon = utils.get_lexicon(source_lex)
print("size of lexicon:", set(words_scr_lexicon).__len__())
#print(len(words_scr_lexicon), len(words_trg_lexicon))

# Cargar vectores Node2Vec español/náhuatl
source_str = "es.n2v"
target_str = "na.n2v"
#source_str = "es.norm.n2v"
#source_str = "en.fst"
source_vec = utils.open_file(source_str)
words_src, source_vec = utils.read(source_vec, is_zipped=False)
# lista de palabras en español del lexicon semilla
eval_src = list(set(words_scr_lexicon))
src_vec = utils.get_vectors(eval_src, words_src, source_vec)
print("source_vec: " + source_str)
# print(src_vec.shape)


#target_str = "it.fst"
target_vec = utils.open_file(target_str)
words_trg, target_vec = utils.read(target_vec, is_zipped=False)
print("target_vec: " + target_str)
#eval_it = list(set(it))
#trg_vec = get_vectors(eval_it, words_it, it_vec)
# print(target_vec.shape)

test_vectors = src_vec

X = tf.constant(test_vectors, dtype=tf.float32)

# Crear Autoencoder

modelo = Autoencoder(dims=test_vectors.shape[1],
                     dim_encoded=350,
                     dropout=1)

# Para construir modelo;
# Si se pasa sólo un vector se debe
# usar método tf.convert_to_tensor
# Salida es dummy
modelo.predict(X).numpy()


# Cargar modelo guardado numpy=0.13671634>
#weights_path = "ae-weights.h5"

modelo.load_weights("ae-weights.h5", by_name=True)


pred = modelo.predict(X).numpy()

top_10 = [utils.get_topk_vectors(pred[_], target_vec)
          for _ in range(pred.shape[0])]


closest = [utils.closest_word_to(top_10[_], words_trg)
           for _ in range(pred.shape[0])]


resultados = {palabra_en: top_10_it for (
    palabra_en, top_10_it) in zip(eval_src, closest)}


gold = utils.gold_dict(words_scr_lexicon, words_trg_lexicon)


p1, p5, p10 = 0, 0, 0
list_en_eval = list(resultados.keys())
hits, not_found = [], []

# Mostrar los candidatos a traducción del lexicón de evaluación
# for palabra in eval_src:
#     print("Traducción de:", palabra)
#     for i, w in enumerate(resultados[palabra]):
#         print("\t", str(i + 1) + ".- " + w)
#     print()


# Medir Precision_at_k
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
