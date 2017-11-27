import numpy as np

dict_file = np.load('npdict_w2v.npy')
npdict_w2v = dict_file.item()
print('found {} words'.format(len(npdict_w2v)))