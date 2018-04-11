
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import utils

from collections import Counter

import time

start_time = time.time()


__author__ = "Olivares Castillo José Luis"

#tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

print("TensorFlow version: {}".format(tf.VERSION))
#print("Eager execution: {}".format(tf.executing_eagerly()))

if tf.test.gpu_device_name():
    print("GPU disponible")


# In[2]:


words_scr_lexicon, words_trg_lexicon = utils.get_lexicon("en-it.test")
#print(len(words_scr_lexicon), len(words_trg_lexicon))


# In[3]:


source_vec = utils.open_file('en.fst')
words_src, source_vec = utils.read(source_vec,is_zipped=False)
eval_src = list(set(words_scr_lexicon))
src_vec = utils.get_vectors(eval_src, words_src, source_vec)
#print(src_vec.shape)


# In[4]:


target_vec = utils.open_file("it.fst")
words_trg, target_vec = utils.read(target_vec,is_zipped=False)
#eval_it = list(set(it))
#trg_vec = get_vectors(eval_it, words_it, it_vec)
#print(target_vec.shape)


# In[5]:


test_vectors = src_vec


# In[6]:


tf.reset_default_graph()
sess = tf.Session()
path="models/en-it/2/"
saver = tf.train.import_meta_graph(path+"model2250.ckpt.meta")
saver.restore(sess,tf.train.latest_checkpoint(path))


# In[7]:


graph = tf.get_default_graph()
X = graph.get_tensor_by_name("input/input_es:0")
kprob = graph.get_tensor_by_name("dropout_prob:0")


# In[8]:


#([n.name for n in graph.as_graph_def().node])


# In[16]:


output_NN = graph.get_tensor_by_name("nah_predicted/BiasAdd:0")
#output_NN = graph.get_tensor_by_name("nah_predicted:0")
#code = graph.get_tensor_by_name("xw_plus_b_2:0")
#print(output_NN)

feed_dict = {X: test_vectors, kprob: 1}
pred = sess.run(output_NN, feed_dict)
#print(pred.shape)


# In[17]:


#get_ipython().run_cell_magic('time', '', 'top_10 = [utils.get_top10_vectors(pred[_], target_vec) for _ in range(pred.shape[0])]')
top_10 = [utils.get_top10_vectors(pred[_],target_vec) for _ in range(pred.shape[0])]

# In[18]:


#get_ipython().run_cell_magic('time', '', 'closest = [utils.closest_word_to(top_10[_], words_trg) for _ in range(pred.shape[0])]')
closest = [utils.closest_word_to(top_10[_], words_trg) for _ in range(pred.shape[0])]

# In[19]:


resultados = {palabra_en: top_10_it for (palabra_en, top_10_it) in zip(eval_src, closest)}


# In[20]:


gold = utils.gold_dict(words_scr_lexicon, words_trg_lexicon)


# In[21]:


#get_ipython().run_cell_magic('time', '', 'p1, p5, p10 = 0, 0, 0\nlist_en_eval = list(resultados.keys())\nhits, not_found = [], []\n\nfor palabra_gold in list_en_eval:\n    for i in gold[palabra_gold]:\n        if i in resultados[palabra_gold]:\n            hits.append(resultados[palabra_gold].index(i))\n    if hits.__len__() > 0:\n        if min(hits) == 0:\n            p1 += 1\n            p5 += 1\n            p10 += 1\n        if min(hits) >= 1 and min(hits) <= 5:\n            p5 += 1\n            p10 += 1\n        if min(hits) > 5 and min(hits) < 10:\n            p10 += 1\n    else:\n        not_found.append(palabra_gold)\n    hits.clear()\n\nlength = list_en_eval.__len__()\nprint("not found:", not_found.__len__(), "-", not_found.__len__() / length, "%")\nprint("P@1:", p1, "\\tP@5:", p5, "\\tP@10:", p10)\nprint("P@1:", p1 / length, "\\tP@5:", p5 /length, "\\tP@10:", p10 / length)')
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
print("not found:", not_found.__len__(), "-", not_found.__len__() / length, "%")
print("P@1:", p1, "\tP@5:", p5, "\tP@10:", p10)
print("P@1:", p1 / length, "\tP@5:", p5 /length, "\tP@10:", p10 / length)
e=time.time()-start_time
print("Time: %02d:%02d:%02d" % (e // 3600, (e % 3600 // 60), (e % 60 // 1)))
