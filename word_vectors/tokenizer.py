'''
Created on 29.11.2017

@author: mkarjanm
'''
import nltk
import re
import string

def tokenize(string_to_tokenize):
    w = nltk.word_tokenize(string_to_tokenize)
    sanitized = []
    is_word = re.compile('[%s]' % re.escape(string.punctuation))
    unwanted = re.compile('[_]')
    for token in w: 
        new_token = is_word.sub(u'', token)
        clean_token = unwanted.sub(u'', token)
        if not new_token == u'':
            sanitized.append(clean_token.lower())   
    return sanitized