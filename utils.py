# -*- coding: utf-8 -*-

import numpy as np
import gzip
from zipfile import ZipFile
from io import BufferedReader
from collections import defaultdict

__author__ = "Olivares Castillo José Luis"


def read(file, vocabulary=None, is_zipped=True, encoding="utf-8", dtype=np.float64):

    words = []

    if is_zipped:
        header = file.readline().strip().decode(encoding).split()
    else:
        header = file.readline().split()
    count = int(header[0])  # if is_zipped else 5000
    dim = int(header[1])  # if is_zipped else 300
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        if is_zipped:
            word, vec = file.readline().decode(encoding).split(" ", 1)
        else:
            try:
                word, vec = file.readline().split(" ", 1)
            except ValueError as e:
                print(e, file.readline().split(" ", 1), i)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=" ", dtype=dtype)
        elif word in vocabulary:
            #for _ in range(vocabulary[word]):
            words.append(word)
            matrix.append(np.fromstring(vec, sep=" ", dtype=dtype))
    file.close()
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


def closest_word_to(top_10, words):
    return [words[index] for index in top_10]


def get_lexicon(source, path=None):
    path = "datasets/dictionaries/" if path is None else path
    if source.__eq__("en-it.train"):
        src, trg = load_lexicon(path + "en-it.train.drive.txt")
    elif source.__eq__("en-it.test"):
        src, trg = load_lexicon(path + "en-it.test.drive.txt")
    elif source.__eq__("en-de.test"):
        src, trg = load_lexicon(path + "en-de.test.txt")
    elif source.__eq__("en-de.train"):
        src, trg = load_lexicon(path + "en-de.train.txt")
    elif source.__eq__("en-es.test"):
        src, trg = load_lexicon(path + "en-es.test.txt")
    elif source.__eq__("en-es.train"):
        src, trg = load_lexicon(path + "en-es.train.txt")
    elif source.__eq__("en-fi.test"):
        src, trg = load_lexicon(path + "en-fi.test.txt")
    elif source.__eq__("en-fi.train"):
        src, trg = load_lexicon(path + "en-fi.train.txt")
    elif source.__eq__("es-na.train"):
        src, trg = load_lexicon(path + "es-na.train.txt")
    elif source.__eq__("es-na.test"):
        src, trg = load_lexicon(path + "es-na.test.true.txt")
    else:
        raise ValueError("{} no encontrado".format(source))
    return (src, trg)


def load_lexicon(source):
    src, trg = [], []
    with open(source, "r") as file:
        for line in file:
            src.append(line.split()[0])
            trg.append(line.split()[1])
    return (src, trg)


def open_file(source,path=None):
    path = "datasets/" if path is None else path
    if source.__eq__("en.norm.fst"):
        return open(path+"en.200k.300d.norm.fst", "r", encoding="utf-8", errors="surrogateescape")
    if source.__eq__("en.fst"):
        return open(path+"en.200k.300d.fst", "r", encoding="utf-8", errors="surrogateescape")
    if source.__eq__("it.fst"):
        return open(path+"en-it/it.200k.300d.fst", "r", encoding="utf-8", errors="surrogateescape")
    if source.__eq__("it.norm.fst"):
        return open(path+"en-it/it.200k.300d.norm.fst", "r", encoding="utf-8", errors="surrogateescape")
    if source.__eq__("it.fst"):
        return open(path+"en-it/it.200k.300d.fst", "r", encoding="utf-8", errors="surrogateescape")
    if source.__eq__("en-wiki"):
        #2519370 vectors
        file = ZipFile(path+"wiki/wiki.en.zip")\
            .open("wiki.en.vec")
    elif source.__eq__("en-wiki-news"):
        file = ZipFile(path+"wiki-news-300d-1M-subword.vec.zip")\
            .open("wiki-news-300d-1M-subword.vec")
    elif source.__eq__("en-crawl"):
        file = ZipFile(path+"crawl-300d-2M.vec.zip")\
            .open("crawl-300d-2M.vec")
    elif source.__eq__("it"):
        file = gzip.open(path+"en-it/cc.it.300.vec.gz")
    elif source.__eq__("de"):
        file = gzip.open(path+"en-de/cc.de.300.vec.gz")
    elif source.__eq__("fi"):
        file = gzip.open(path+"en-fi/cc.fi.300.vec.gz")
    elif source.__eq__("es"):
        file = gzip.open(path+"en-es/cc.es.300.vec.gz")
    elif source.__eq__("es.n2v"):
        file = open("es.node2vec.embeddings")
    elif source.__eq__("es.norm.n2v"):
        file = open("es.node2vec.norm.embeddings")
    elif source.__eq__("na.n2v"):
        file = open("na.node2vec.embeddings")
    elif source.__eq__("na.norm.n2v"):
        file = open("na.node2vec.norm.embeddings")
    else:
        raise ValueError("{} no encontrado".format(source))
    return BufferedReader(file)


def get_vectors(lexicon, words, embeddings, dtype='float'):
    matrix = np.empty((len(lexicon), embeddings.shape[1]), dtype=dtype)
    for i in range(len(lexicon)):
        if lexicon[i] in words:
            matrix[i] = embeddings[words.index(lexicon[i])]
    return np.asarray(matrix, dtype=dtype)


def next_batch(x, y, step, batch_size):
    ix = batch_size * step
    iy = ix + batch_size
    return x[ix:iy], y[ix:iy]


def get_top10_vectors(vector, matrix, kind="quicksort"):
    unsorted = ((np.matmul(vector, matrix.T) / (np.linalg.norm(vector) *
                                                np.sqrt(np.einsum('ij,ij->i', matrix, matrix)))))
    unsorted = np.argsort(unsorted, kind=kind)
    distances = unsorted[::-1][:10]
    del unsorted
    return distances


def gold_dict(list_src, list_trg):
    # Lista de pares traducción
    pares_eval = list(zip(list_src, list_trg))

    # Diccionario con listas en su valor
    gold = defaultdict(list)

    # Se genera una lista de traducciones gold standard para cada palabra del idioma fuente
    for palabra_src, palabra_trg in pares_eval:
        gold[palabra_src].append(palabra_trg)

    # Se eliminan variables innecesarias
    del pares_eval

    # Se hace cast al defaultdict y se retorna un diccionario de python.
    return dict(gold)
