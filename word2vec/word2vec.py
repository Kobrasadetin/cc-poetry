# -*- coding: utf8 -*-
'''
Created on 5.12.2017

@author: mkarjanm

This module can be used to acquire the pretrained word vectors from word2vec.
This is a cleaned and untested version of my original code, so there could be bugs.
'''
import nltk
import numpy as np
from gensim.models import Word2Vec
from collections import Counter
import os

def find_vectors_for_tokens(tokens, model_filename= os.path.join("K:","Ohjelmointi", "googleModel")):
    '''returns tuple (dictionary of {tokens:vectors}, unmatched tokens)'''
    print('loading w2v model...')
    model = Word2Vec.load(model_file)
    print('finding vectors from w2v...')
    npdict_w2v={}
    missing = []
    counter=0
    for w in tokens:
        counter+=1
        try:
            w2v_vector = model[w]
            npdict_w2v[w] = w2v_vector
        except KeyError as err:
            missing.append(w)
        if counter%1000 == 0:
            print('{}/{} vectorized'.format(counter, len(tokens), end='\r'))
    print('done.')
    return npdict_w2v, missing

if __name__ == '__main__':
    w = np.load('missing_tokens.npy')
    print (len(w))
    a =  np.load('npdict_w2v.npy').item()
    print(len(a))
    exit()
    model_file = os.path.join("K:","Ohjelmointi", "googleModel")
    with open('20k.txt', 'r', encoding='utf8') as f:
        words = nltk.word_tokenize(f.read())
    words.extend(np.load('missing_tokens.npy'))
    print( len(words))
    npdict_w2v, missing = find_vectors_for_tokens(words)
    np.save('npdict_w2v.npy', npdict_w2v)
    np.save('missing.npy', missing)