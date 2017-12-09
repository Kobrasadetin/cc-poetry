# -*- coding: utf8 -*-
'''
Created on 29.11.2017

@author: mkarjanm
'''
import random
import numpy as np
import csv
import logging
from word_vectors.vectorizer import Vectorizer
from word_vectors import tokenizer
from numba.tests.npyufunc.test_ufunc import dtype

log = logging.getLogger("batch_manager")


def read_csv(filename, column, min_length=10):
  '''returns the defined column as list from a csv file
  leaves out rows where column content is shorter than min_length'''
  arr = []
  with open(filename, 'r', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        if (len(row[column]) >= min_length):
            arr.append(row[column])
  return arr

class BatchManager:
  
    def __init__(self, batch_size=128, sequence_length=10, vectorizer_instance=Vectorizer()):
        self.vectorizer = vectorizer_instance
        self.context_arrays=[]
        self.sequences=[]
        self.sequence_indexes=[]
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sequence_counter = 0
        self.all_tokens = set()
        
    def read_csv(self, filename, column, limits = slice(0, 10000)):      
        log.info('opening csv {}'.format(filename))     
        contents = read_csv(filename, column)
        counter = 0   
        counts_all = 0
        counts_zero = 0     
        for context in contents[limits]:         
            tokenized = tokenizer.tokenize(context)
            if tokenized != None:
                for token in tokenized:
                    self.all_tokens.add(token)
                vectorized, miss_indexes = self.vectorizer.vectorize_tokens(tokenized)            
                '''vectorizer returns numpy arrays of varying length'''
                self.context_arrays.append(vectorized)
                zerocount = 0            
                for i in range(np.shape(vectorized)[0] - self.sequence_length + 1):
                    '''only include samples where the last word is not equal to a zero vector'''
                    if miss_indexes[i+self.sequence_length - 1] == 0:
                        self.sequences += [(vectorized[i:i+self.sequence_length], miss_indexes[i:i+self.sequence_length])] 
                        counts_all += 1
                    else:
                        zerocount+=1    
                counts_zero += 1 if zerocount>0 else 0
                counter += 1
                if (counter % 1000 == 0): print('{:06d}/{:06d},\t {} training vectors'.format(counter, len(contents), counts_all), end='\r')
        '''shuffle sequences'''  
        print("done.")      
        random.shuffle(self.sequences)               
    
    def all_word_tokens(self):
        '''returns a set of all the tokens in the managed batches'''
        return self.all_tokens         
                
    def next_batch(self):
        '''returns the next batch of random sequences'''
        new_batch = self.sequences[self.sequence_counter : self.sequence_counter + self.batch_size]
        self.sequence_counter += self.batch_size
        
        '''reshuffle sequences when all have been processed'''
        if (self.sequence_counter >= (len(self.sequences) - self.batch_size)):
            random.shuffle(self.sequences)
            self.sequence_counter = 0
             
        return new_batch
    
