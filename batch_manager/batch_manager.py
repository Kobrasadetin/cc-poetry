'''
Created on 29.11.2017

@author: mkarjanm
'''
import random
import numpy as np
import csv
from word_vectors.vectorizer import Vectorizer
from word_vectors import tokenizer
from numba.tests.npyufunc.test_ufunc import dtype


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
  
    def __init__(self, batch_size=128, sequence_length=10):
        self.vectorizer = Vectorizer()
        self.context_arrays=[]
        self.sequences=[]
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.sequence_counter = 0
        
    def read_csv(self, filename, column):
        contents = read_csv(filename, column)        
        for context in contents:
            tokenized = tokenizer.tokenize(context)           
            vectorized = self.vectorizer.vectorize_tokens(tokenized)
            '''vectorizer returns numpy arrays of varying length'''
            self.context_arrays.append(vectorized)
            for i in range(np.shape(vectorized)[0] - self.sequence_length + 1):
                self.sequences += [vectorized[i:i+self.sequence_length]]
        '''shuffle sequences'''
        random.shuffle(self.sequences)
                
    def next_batch(self):
        '''selects the next batch of random sequences'''
        new_batch = np.array(self.sequences[self.sequence_counter : self.sequence_counter + self.batch_size]) 
        self.sequence_counter += self.batch_size
        
        '''reshuffle sequences when all have been processed'''
        if (self.sequence_counter >= (len(self.sequences) - self.batch_size)):
            random.shuffle(self.sequences)
            self.sequence_counter = 0
             
        return new_batch
        
    
    