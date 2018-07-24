from __future__ import division
import matplotlib.pyplot as plt
from tsne import tsne
import networkx as nx
import numpy as np
from math import fabs, log

def plot_vecs(Z,ids, color='blue', form='o', show=True):
    Z = tsne(Z, 2)
    r=0
    plt.scatter(Z[:,0],Z[:,1], marker=form, c=color)
    for label,x,y in zip(ids, Z[:,0], Z[:,1]):
        plt.annotate(label, xy=(x,y), xytext=(-1,1), textcoords='offset points', ha='center', va='bottom')
        r+=1
    if show == True:
        plt.show()
    else:
        pass


def g2vec(G):
	#words = open('target_words.txt','r').read().decode('utf-8').split('\n')
	v = {x:y for y,x in enumerate(G.nodes())}
	A = nx.to_numpy_matrix(G)
	#words = v.keys() #set(words).intersection(set(G.nodes()))

	#A1 = np.zeros((len(words), A.shape[0]))
	#ids = {}
	#for i,w in enumerate(words):
	#	try:
	#		A1[i] = A[v[w]]
	#		ids[w] = i
	#	except:
	#		pass

	return np.array(A), v

def cos(x,y):
	normx = np.linalg.norm(x)
	normy = np.linalg.norm(y)
	if normx != 0 and normy != 0:
		return fabs(np.dot(x,y)) / (normx*normy)
	else:
		return 0.0

def normalize(A):
	An = np.zeros(A.shape)
	for i,v_a in enumerate(A):
		norm = np.linalg.norm(v_a)
		if norm != 0.0:
			An[i] = v_a/norm
	return An

def log2(x):
    if x == 0:
        return 0.0
    else:
        return log(x,2)
