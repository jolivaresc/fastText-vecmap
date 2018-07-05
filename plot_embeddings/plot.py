import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def read(path,count=10):
    matrix = np.empty((count, 128), dtype=np.float)
    words = []
    f = open(path, "r")
    for i in range(count):
        w, vec = f.readline().split(" ", 1)
        words.append(w)
        matrix[i] = np.fromstring(vec, sep=" ", dtype=np.float)
    return (words, matrix)


palabra_es, matrix_es = read("es.embeddings")
palabra_na, matrix_na = read("na.embeddings")


method = TSNE

es = method(n_components=2,method="barnes_hut").fit_transform(matrix_es)
na = method(n_components=2,method="barnes_hut").fit_transform(matrix_na)

fig, ax = plt.subplots()
ax.scatter(es[:, 0], es[:, 1], marker="o")
ax.scatter(na[:, 0], na[:, 1], marker="d",c="r")
#ax2.scatter(na[:, 0], na[:, 1], c="r", marker="d")

for i, (text_es, text_na) in enumerate(zip(palabra_es, palabra_na)):
    ax.annotate(text_es, (es[i, 0], es[i, 1]))
    ax.annotate(text_na, (na[i, 0], na[i, 1]))

ax.grid()
ax.set_title("Español")

# plt.scatter(reduction[:,0],reduction[:,1])

plt.show()