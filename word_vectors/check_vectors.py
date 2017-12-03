import numpy as np
from matplotlib.mlab import PCA
import word_vectors.precalc_eigenvector
import random

dict_file = np.load('npdict_w2v.npy')
npdict_w2v = dict_file.item()
print('found {} words'.format(len(npdict_w2v)))
print('vector size {}'.format(len(next (iter (npdict_w2v.values())))))

data = []
for key, value in npdict_w2v.items():
    data.append(value)
data = np.array(data)
pca_results = PCA(data)

print('furthest euclidean dist')
#for i in range(len(npdict_w2v))

print('pca:')
print('fracs:')
print(pca_results.fracs)
print('----------------- first transform axis')
print(repr(pca_results.Wt[0]))

print('orig')
avg = np.average(data[:,0])
sqrs1 = np.sqrt(np.sum(np.power(data[:,0] - avg, 2)))
avg = np.average(data[:,1])
sqrs2 = np.sqrt(np.sum(np.power(data[:,1] - avg, 2)))
avg = np.average(data[:,2])
sqrs3 = np.sqrt(np.sum(np.power(data[:,2] - avg, 2)))
print('first dim mean sqr dev {}'.format(sqrs1))
print('2nd dim mean sqr dev   {}'.format(sqrs2))
print('3rd dim mean sqr dev   {}'.format(sqrs3))

print('transformed')
eig = np.matmul(data, pca_results.Wt[0])
avg = np.average(eig)
sqrs1 = np.sqrt(np.sum(np.power(eig - avg, 2)))
print('first eig mean sqr dev {}'.format(sqrs1))
eig = np.matmul(data, pca_results.Wt[1])
avg = np.average(eig)
sqrs1 = np.sqrt(np.sum(np.power(eig - avg, 2)))
print('2nd eig mean sqr dev   {}'.format(sqrs1))
eig = np.matmul(data, pca_results.Wt[2])
avg = np.average(eig)
sqrs1 = np.sqrt(np.sum(np.power(eig - avg, 2)))
print('3rd eig mean sqr dev   {}'.format(sqrs1))

words=[]
for key, value in npdict_w2v.items():
    val = np.matmul(value, pca_results.Wt[0])
    if abs(val) > 2 or abs(val) < 0.05:
        words += [(key, np.matmul(value, pca_results.Wt[0]))]

words.sort(key=lambda tup: tup[1])
print (words)

words=[]
for key, value in npdict_w2v.items():
    val = np.matmul(value, pca_results.Wt[1])
    if abs(val) > 1.7 or abs(val) < 0.05:
        words += [(key, np.matmul(value, pca_results.Wt[1]))]

words.sort(key=lambda tup: tup[1])
print (words)
    


