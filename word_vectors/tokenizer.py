# -*- coding: utf8 -*-
'''
Created on 29.11.2017

@author: mkarjanm
'''
import nltk
import re
import string

substitutes = [('―','-'),('٭',''),('ﬁ','fi'),('`', "'"),('“', '"'),('”','"')]
removed_punctuation = "#$%&()*+-:;<=>@[\]^_{|}~"
allowed_punctuation = "!\".,?'/"
pattern_subs = [("'m", " am"),("can't", "can not"),("won't", "will not"),("n't", " not"),("'ll", " will"), ("'d"," would"), ("'ll", " will"), ("'ve", " have"), ("'s", " is"), ("'re", " are") ]


def tokenize(string_to_tokenize, patterns=True):
    '''this tokenizer returns lowercase words
    currently we don't remove most of the punctuation
    '''
    for val1, val2 in substitutes:
        string_to_tokenize = string_to_tokenize.replace(val1, val2)
    if (patterns):
        for val1, val2 in pattern_subs:
            string_to_tokenize = string_to_tokenize.replace(val1, val2)
    w = nltk.word_tokenize(string_to_tokenize)
    sanitized = []
    #unwanted = re.compile('[%s]' % removed_punctuation)
    
    for token in w: 
        #new_token = unwanted.sub(u'', token)
        new_token = token
        for val1, val2 in substitutes:
            new_token = new_token.replace(val1, val2)
        if not new_token == u'':
            sanitized.append(new_token.lower())   
    return sanitized