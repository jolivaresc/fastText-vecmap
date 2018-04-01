# -*- coding: utf-8 -*-

import numpy as np

__author__ = "Olivares Castillo José Luis"


def read(file,vocabulary=None,is_zipped=True,dtype=np.float32):
    
    words = []
    
    if is_zipped:
        header = file.readline().strip().decode().split()
        
    count = int(header[0]) if is_zipped else 5000
    dim = int(header[1]) if is_zipped else 300
    matrix = np.empty((count,dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        if is_zipped:
            word, vec = file.readline().decode().split(" ", 1)
        else:
            try:
                word, vec = file.readline().split(" ", 1)
            except ValueError as e:
                print(e, file.readline().split(" ", 1),i)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=" ", dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec,sep=" ", dtype=dtype))
    file.close()
    return (words,matrix) if vocabulary is None else (words,np.array(matrix,dtype=dtype))
