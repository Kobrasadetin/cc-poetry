'''
Created on 29.11.2017

@author: mkarjanm
'''
import numpy as np
import os

class Vectorizer:
    def __init__(self):
        self.dict_file = np.load(os.path.join('word_vectors','npdict_w2v.npy'))
        self.npdict_w2v = self.dict_file.item()
        self.size_of_vec = len(next (iter (self.npdict_w2v.values())))
        #300 when using word2vec
    def vectorize_tokens(self, tokens):
        '''vectorizes a list of tokens
        returns a numpy array of size [len(tokens), size_of_vec]'''
        vec_array = np.empty([len(tokens), self.size_of_vec], dtype=np.float32)
        for i in range(len(tokens)):
            vec = np.zeros(self.size_of_vec)
            try:
                vec = self.npdict_w2v[tokens[i]]
            except KeyError:
                pass
            vec_array[i] = vec
        return vec_array