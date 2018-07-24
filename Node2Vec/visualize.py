from pickle import load
from utils import plot_vecs
import numpy as np

def visual(name,vector):
	M,voc = load(open(vector,'r'))

	voc = {v:u.decode('utf-8') for u,v in voc.iteritems()}
	#print voc

	f = open(name,'r').read().strip().split('\n')
	f.pop(0)

	V = np.zeros((M.shape[0],300))
	for v in f:
		todo = v.split()
		id = int(todo[0])
		todo.pop(0)
		V[id] =  np.array([float(x) for x in todo])

	return V,voc

V1,voc1 = visual('esp.emd','DSM-esp.p')
V2,voc2 = visual('nah.emd','DSM-nah.p')
W = np.concatenate((V1,V2))
print( W.shape)
plot_vecs(W,voc1.values()+voc2.values())
