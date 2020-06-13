import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

import pickle

import os
import glob

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    #remove HTML tags
    text = BeautifulSoup(review, "html.parser").get_text()
    
    #convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #split string into words
    words = text.split()
    
    #remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    
    #stem
    words = [PorterStemmer().stem(w) for w in words]
    
    return words

def convert_and_pad(word_dict, sentence, pad=500):
    #we will use 0 to represent the 'no word' category
    NOWORD = 0
    
    #we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    INFREQ = 1
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)