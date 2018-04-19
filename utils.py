# -*- coding: utf-8 -*-


from collections import defaultdict
from io import BufferedReader
from zipfile import ZipFile

import numpy as np

import gzip

__author__ = "Olivares Castillo José Luis"


def read(file, vocabulary=None, is_zipped=True, encoding="utf-8", dtype=np.float64):
    """Función para leer datasets en formato:
    palabra vector
    por cada línea

    Parameters:
    ----------
    file : {io.TextIOWrapper}
        Apuntador al archivo a leer
    vocabulary : {list}, optional
        Lista con pares de palabras: sirve para filtrar del dataset
        sólo las palabras que se encuentren en la lista `vocabulary` 
        (the default is None, which [No se carga ningún diccionario])
    is_zipped : {bool}, optional
        Bandera que indica si el archivo está en formato [zip, gz] 
        o en texto simple y poder leerlo según el formato necesario
        (the default is True, which [El archivo está en formato comprimido])
    encoding : {str}, optional
        Tipo de encoding de los archivos
        (the default is "utf-8", which [default_description])
    dtype : {numpy format}, optional
        Tipo de dato de los vectores (the default is np.float64,
        which [formato float64])

    Returns
    -------
    [list],[numpy.ndarray]
        Retorna una lista con las palabras y una matriz del dataset.
    """

    words = []

    # Si el archivo está comprimido, usar el método decode para poder leerlo.
    if is_zipped:
        header = file.readline().strip().decode(encoding).split()
    else:
        header = file.readline().split()

    # Se lee la cabecera del archivo. Indica el número total embeddings y
    # su dimensionalidad.
    count = int(header[0])  # if is_zipped else 5000
    dim = int(header[1])  # if is_zipped else 300

    # Se crea una matriz de dimension (count,dim) si se especifica un vocabulario
    # sino se crea una lista.
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []

    # Ciclo para leer el archivo línea por línea `count` veces.
    for i in range(count):
        if is_zipped:
            word, vec = file.readline().decode(encoding).split(" ", 1)
        else:
            # try:
            word, vec = file.readline().split(" ", 1)
            # except ValueError as e:
            #    print(e, file.readline().split(" ", 1), i)
        # Si no se indica un vocabulario se lee el dataset completo
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=" ", dtype=dtype)
        # Si se indica un vocabulario, sólo se cargan las palabras del dataset necesarias.
        elif word in vocabulary:
            # for _ in range(vocabulary[word]):
            words.append(word)
            matrix.append(np.fromstring(vec, sep=" ", dtype=dtype))

    # Cerrar apuntador al archivo
    file.close()

    # Retorna lista con palabras y la matriz de embeddings
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


def get_lexicon(source, path=None):
    """Función para cargar lexicones de entrenamiento/pruebas

    Parameters:
    ----------
    source : {str}
        Lexicon que se va a cargar
    path : {str}, optional
        Indica la ruta en donde se ubican los lexicones
        (the default is None, which [ruta por defecto])

    Raises
    ------
    ValueError
        Si no encuentra el lexicon solicitado, muestra un error de que no lo encontró
        o de que el nombre es inválido.

    Returns
    -------
    list
        Lista con los pares de palabra del lexicon indicado.
    """

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

    # Si no encuentra el archivo o en nombre es inválido muestra un error.
    else:
        raise ValueError("{} no encontrado".format(source))

    # Retorna pares de palabra en listas separadas.
    return (src, trg)


def load_lexicon(source):
    """Función auxiliar de `open_file`. Lee el archivo indicado en `source` y lo retorna
    en dos listas. El lexicon está en formato:
    palabra1 palabra2
    por cada línea.

    Parameters:
    ----------
    source : {str}
        Ruta del lexicon a leer
    Returns
    -------
    list
        Retorna dos listas con los pares de palabra.
    """

    # listas para cargar las palabras del lexicon
    src, trg = [], []

    # Se abre el archivo
    with open(source, "r") as file:
        # Ciclo para leer el archivo línea por línea
        for line in file:
            # Se divide la línea
            # palabra1 palabra2
            # para añadirlas a las listas `src`, `trg`
            src.append(line.split()[0])
            trg.append(line.split()[1])

    # Listas con los pares de palabras por separado.
    return (src, trg)


def open_file(source, path=None):
    """Función para leer datasets. Los datasets están en formato [zip, gz] o simple. 

    Parameters:
    ----------
    source : {str}
        Dataset a leer.
    path : {str}, optional
        Ruta del dataset (the default is None, which [Ruta por defecto])

    Raises
    ------
    ValueError
        Si no encuentra el dataset solicitado, muestra un error de que no lo encontró
        o de que el nombre es inválido.

    Returns
    -------
    io.TextIOWrapper
        Apuntador al dataset requerido.
    """

    # Si no se especifica una nueva ruta se utilza una por defecto.
    path = "datasets/" if path is None else path

    # Se carga en dataset indicado en `source`.

    # Si el dataset está en formato simple i.e: no comprimido se retorna
    # el apuntador al archivo.
    if source.__eq__("en.norm.fst"):
        return open(path + "en.200k.300d.norm.fst", "r", encoding="utf-8", errors="surrogateescape")
    elif source.__eq__("it.norm.fst"):
        return open(path + "en-it/it.200k.300d.norm.fst", "r", encoding="utf-8", errors="surrogateescape")
    elif source.__eq__("en.fst"):
        return open(path + "en.200k.300d.fst", "r", encoding="utf-8", errors="surrogateescape")
    elif source.__eq__("it.fst"):
        return open(path + "en-it/it.200k.300d.fst", "r", encoding="utf-8", errors="surrogateescape")

    elif source.__eq__("es.norm.n2v"):
        return open(path + "es-na/es.node2vec.norm.embeddings")
    elif source.__eq__("na.norm.n2v"):
        return open(path + "es-na/na.node2vec.norm.embeddings")
    elif source.__eq__("es.n2v"):
        return open(path + "es-na/es.node2vec.embeddings")
    elif source.__eq__("na.n2v"):
        return open(path + "es-na/na.node2vec.embeddings")

    # Dataset comprimidos.
    elif source.__eq__("en-wiki"):
        # 2519370 vectors
        file = ZipFile(path + "wiki/wiki.en.zip")\
            .open("wiki.en.vec")
    elif source.__eq__("en-wiki-news"):
        file = ZipFile(path + "wiki-news-300d-1M-subword.vec.zip")\
            .open("wiki-news-300d-1M-subword.vec")
    elif source.__eq__("en-crawl"):
        file = ZipFile(path + "crawl-300d-2M.vec.zip")\
            .open("crawl-300d-2M.vec")
    elif source.__eq__("it"):
        file = gzip.open(path + "en-it/cc.it.300.vec.gz")
    elif source.__eq__("de"):
        file = gzip.open(path + "en-de/cc.de.300.vec.gz")
    elif source.__eq__("fi"):
        file = gzip.open(path + "en-fi/cc.fi.300.vec.gz")
    elif source.__eq__("es"):
        file = gzip.open(path + "en-es/cc.es.300.vec.gz")

    # Si no encuentra el archivo o en nombre es inválido muestra un error.
    else:
        raise ValueError("{} no encontrado".format(source))

    # Retorna apuntador al archivo.
    return BufferedReader(file)


def get_vectors(lexicon, words, embeddings, dtype='float'):
    """Función que busca los embeddings que corresponden a una lista de palabras
    en un dataset dado.

    Parameters:
    ----------
    lexicon : {list}
        Lista que indica las palabras necesarias para buscar el vector que le
        corresponde dentro del dataset.
    words : {list}
        Lista con todas las palabras que existen dentro del dataset dado.
    embeddings : {numpy.ndarray}
        Matriz con todos los embeddings que existen dentro del dataset dado
    dtype : {str}, optional
        Tipo de dato de los embeddings (the default is 'float')

    Returns
    -------
    numpy.ndarray
        Matriz con los embeddings que se especificaron en `lexicon`
    """

    # Se crea una matriz de dimensionalidad `((len(lexicon), embeddings.shape[1])`
    matrix = np.empty((len(lexicon), embeddings.shape[1]), dtype=dtype)

    # Ciclo para llenar la matriz
    for i in range(len(lexicon)):
        # Busca las palabras indicadas en `lexicon` dentro de `words`
        # para obtener su embedding.
        if lexicon[i] in words:
            matrix[i] = embeddings[words.index(lexicon[i])]

    # Retorna una matriz con los embeddings especificados en `lexicon`.
    return np.asarray(matrix, dtype=dtype)


def next_batch(x, y, step, batch_size):
    """Función para generar batches a partir de una matriz.

    Parameters:
    ----------
    x : {numpy.ndarray}
        Matriz x.
    y : {numpy.ndarray}
        Matriz y.
    step : {int}
        Número de Batch
    batch_size : {int}
        Tamaño del batch
    Returns
    -------
    numpy.ndarray
        Retorna un batch de las matrices x,y
    """

    # Se generan los batches
    ix = batch_size * step
    iy = ix + batch_size

    # Retorna el batch correspondiente a `step`
    return x[ix:iy], y[ix:iy]


def get_topk_vectors(vector, matrix, k=10, kind="quicksort"):
    """Función que mide la similitud coseno entre `vector` y `matrix` y las ordena
    de mayor a menor. 
    Los índices que retorna los obtiene de `matrix` y corresponden
    a los vectores más cercanos entre `vector` y `matrix`.

    Parameters:
    ----------
    vector : {numpy.ndarray}
        Vector a medir.
    matrix : {numpy.ndarray}
        Matriz de embeddings.
    kind : {str}, optional
        Algoritmo de ordenamiento.
        (the default is "quicksort", puede utilizarse: {‘quicksort’, ‘mergesort’, ‘heapsort’})

    Returns
    -------
    list
        Lista con los índices de `matrix` más cercanos a `vector`.
    """

    # Formula para obtener la similitud coseno de manera vectorial entre
    #  un vector y una matriz.
    unsorted = ((np.matmul(vector, matrix.T) / (np.linalg.norm(vector) *
                                                np.sqrt(np.einsum('ij,ij->i', matrix, matrix)))))

    # Se ordena las distancias.
    unsorted = np.argsort(unsorted, kind=kind)

    # El algoritmo hacer ordenamiento ascendente, por lo que se invierte `[::-1]`
    # para que quede en orden descendente y se seleccionan los 10 primeros
    # más cercanos `[:10]`.
    distances = unsorted[::-1][:10]

    # Se eliminan variables innecesarias.
    del unsorted

    # Se retorna una lista con los índices de `matrix` más cercanos a `vector`
    return distances


def closest_word_to(top_10, words):
    """Función que retorna lista de palabras según los índices en `top_10`

    Parameters:
    ----------
    top_10 : {list}
        Lista con índices que se utilizan para buscar en `words`
    words : {list}
        Lista con palabras.
    Returns
    -------
    list
        Lista de palabras según los índices de `words`
    """

    return [words[index] for index in top_10]


def gold_dict(list_src, list_trg):
    """Función que genera un diccionario de palabras y sus traducciones 
    gold-standard.

    Parameters:
    ----------
    list_src : {list}
        Lista de palabras en idioma origen
    list_trg : {list}
        Lista de palabras que son traducciones gold-standard
    Returns
    -------
    dict
        Diccionario donde su llave es la palabra en idioma origen y su valor es una lista
        con sus traducciones 
    """

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
