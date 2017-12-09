# -*- coding: utf8 -*-
'''
Created on 29.11.2017

@author: mkarjanm
'''
import nltk
import re
import string

newline_symbol = ' -newline- '
internet_address_symbol = ' -website- '
substitutes = [('//', newline_symbol), ('―','-'),('٭',''),('ﬁ','fi'),('`', "'"),('“', '"'),('”','"'),('‘',"'")]
remove_if_found_punctuation = re.escape("*^{|}~¡ãã¨î¾µï\¬¹±+§äöå")
# removed_punctuation = "#$%&()*+-:;<=>@[\]^_{|}~"
# allowed_punctuation = "!\".,?'/"
pattern_subs = [("'m", " am"),("can't", "can not"),("won't", "will not"),("n't", " not"),("'ll", " will"), ("'d"," would"), ("'ll", " will"), ("'ve", " have"), ("'s", " is"), ("'re", " are") ]


def tokenize(string_to_tokenize, patterns=False):
    '''this tokenizer returns lowercase words
    currently we don't remove most of the punctuation
    BUT we rturn None if there's any unallowed chartacters
    '''
    string_to_tokenize = string_to_tokenize.replace('\n', newline_symbol)
    for val1, val2 in substitutes:
        string_to_tokenize = string_to_tokenize.replace(val1, val2)
    if (patterns):
        for val1, val2 in pattern_subs:
            string_to_tokenize = string_to_tokenize.replace(val1, val2)
    w = nltk.word_tokenize(string_to_tokenize)
    sanitized = []
    unwanted = re.compile('[%s]' % remove_if_found_punctuation)
    
    for token in w:
        if unwanted.search(token) != None:
            return None
        for val1, val2 in substitutes:
            token = token.replace(val1, val2)
        if token[0:2] == '//':
            token = internet_address_symbol
        if not token == u'':
            sanitized.append(token.lower())   
    return sanitized