'''
Created on 29.11.2017

@author: mkarjanm
'''
import numpy as np
import os
import bisect
import word_vectors.precalc_eigenvector

class Vectorizer:
    def __init__(self, path='word_vectors', file='npdict_w2v.npy'):
        self.dict_file = np.load(os.path.join(path, file))
        self.npdict_w2v = self.dict_file.item()
        self.size_of_vec = len(next (iter (self.npdict_w2v.values()))) #300 when using word2vec
        
        eigenvect = word_vectors.precalc_eigenvector.get_largest_eigenvector()
        self.valuessorted = [(np.matmul(value, eigenvect), key, value) for key, value in self.npdict_w2v.items()]
        self.valuessorted.sort(key=lambda tup: tup[0])
        
        self.dict_tuples = [(value, token) for _, token, value in self.valuessorted]  
        self.vectors = [value for _, _, value in self.valuessorted]       
        self.vectors = np.array(self.vectors)
        
        self.miss_dictionary = {None:None}
                  
        
    def vectorize_tokens(self, tokens):
      
        '''vectorizes a list of tokens
        returns a numpy array of size [len(tokens), size_of_vec]'''
        vec_array = np.empty([len(tokens), self.size_of_vec], dtype=np.float32)
        miss_indexes = np.empty([len(tokens), 1], dtype=np.int32)
        for i in range(len(tokens)):
            miss_index = 0
            vec = np.zeros(shape=self.size_of_vec)
            try:
                vec = self.npdict_w2v[tokens[i]]
            except KeyError: 
                try:
                    miss_index = self.miss_dictionary[tokens[i]]
                except KeyError:
                    next_index = len(self.miss_dictionary)
                    self.miss_dictionary[tokens[i]] = next_index
                    miss_index = next_index
            miss_indexes[i] = miss_index
            vec_array[i] = vec
        return vec_array, miss_indexes
      
    def get_vector(self, token):
        vec = np.zeros(shape=self.size_of_vec)
        try:
            vec = self.npdict_w2v[token]
        except KeyError: 
            pass
        return vec
      
    def contains_token(self, token):
        '''returns true if token has a vector representation'''
        if token in self.npdict_w2v:
            return True
        return False
      
    def all_vectors(self):
        '''returns word vectors in a numpy array'''
        return self.vectors
      
    def all_tuples(self):
        '''returns word vectors in a list of tuples (vector, string)'''
        return self.dict_tuples
    
    def tuple_at_index(self, idx):
        '''returns word vectors in a numpy array'''
        return self.dict_tuples[idx]
      
    def find_nearest(self, value):
        '''returns nearest val, sorted by largest '''
        '''
        idx = bisect(self.values, (value, None))
        if idx > 0 and (idx == len(self.values) or abs(value - self.values[idx-1]) < abs(value - self.values[idx])):
            return self.values[idx-1]
        else:
            return self.values[idx]'''
        best = 1.0e100
        closest = None
        if (np.shape(value) != (300,)):
            raise ValueError('word vector shape != (300,) in vectorizer.find_nearest (shape {})'.format(np.shape(value)))        
        for key, dict_val in self.npdict_w2v.items():
            dist = np.linalg.norm(value - dict_val)
            if dist < best:
                best = dist
                closest = key
        return closest     

'''running as main starts a vector finder loop'''
if __name__=='__main__':
    v = Vectorizer(path='../word_vectors')
    while(True):
        w = input()
        print(v.find_nearest, v.vectorize_tokens([w])[0])