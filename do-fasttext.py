# -*- coding: utf-8 -*-

"""Buscar palabras de w2v en fasttext

para ejecutar:

python do-fasttext.py DATASET_W2V DATASET_FASTTEXT NEW_DATASET_FASTTEXT
"""

import utils
import sys

__author__ = "Olivares Castillo José Luis"

#print(sys.argv)
#sys.exit(-1)

words = []

#with open("datasets/en-it/it.200k.300d.embeddings.w2v", "r") as f:
with open(sys.argv[1], "r", encoding="utf-8", errors="surrogateescape") as f:
    #print("reading " + sys.argv[1])
    header_w2v = f.readline().split()
    #print(header_w2v)

    for line in range(int(header_w2v[0])):
        w, _ = f.readline().split(" ", 1)
        words.append(w)
    #print("size vocab w2v: " + str(len(words)))

#print("\nreading fasttext...")
# = "utf-8"
#fasttext_file = utils.open_file("it")
fasttext_file = utils.open_file(sys.argv[2])
header_fst = fasttext_file.readline().decode().split()
#print(header_fst)

#print("\nwriting fasttext vectors in " + sys.argv[3])
#with open("datasets/en-it/it.tmp.fst", "w") as f:
with open(sys.argv[3], "w") as f:
    for i in range(int(header_fst[0])):
        w, _ = fasttext_file.readline().decode().split(" ", 1)
        if w in words:
            f.write(w + " " + "".join(map(str, _)))

#print("\nFinished")
#2519374
