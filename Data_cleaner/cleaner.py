#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import razdel
from collections import Counter
from nltk.tokenize import sent_tokenize
from gensim.summarization.textcleaner import split_sentences
from sklearn.feature_extraction.text import CountVectorizer
import logging
logging.basicConfig(filename="data/clear_data/cleaner.log", level=logging.INFO)


# In[2]:


files = ['films_1_6500.csv', 'films_2_6500.csv', 'dom2.csv', 'kinopoisk.csv']

data = pd.concat([
                  pd.read_csv('data/parse_data/'+file, 
                              usecols=['text'], nrows=50000).dropna()
                  for file in files
])


# In[3]:


import re

logging.info("Informational message")


znaks = r"[,:!()?]"
def tokenize(string):
    string = re.sub(r"[^A-Za-zА-Яа-я0-9()\-,!?\'\`’:]", " ", string)
    string = re.sub(r"’|`", "'", string)
    string = re.sub(znaks, lambda x: f' {x.group()} ', string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    tokens = [token.text for token in razdel.tokenize(string)]
    tokenize.all_count_token += len(tokens)
    return tokens

tokenize.all_count_token = 0


# In[4]:


get_ipython().run_cell_magic('time', '', 'cv = CountVectorizer(tokenizer=tokenize)\ncv.fit(data.text)')


# In[5]:


all_words = cv.transform(data.text)
words_sum = all_words.sum(axis=0)


# In[6]:


vocab = [(word, words_sum[0, idx]) for word, idx in cv.vocabulary_.items()]
vocab = sorted(vocab, key = lambda x: x[1])
vocab.extend([('<UNK>', 'unknown'), ('<S>', 'start'), ('</S>', 'end')])
vocab.reverse()


# In[7]:


logging.info(f' count tokens = {tokenize.all_count_token}')
logging.info(f' vocab size = {len(vocab)}')


# In[8]:


vocab_txt = 'data/clear_data/vocab.txt'
with open(vocab_txt,'a') as f:
    for token, token_count in vocab:
        f.write(token + '\n')


# In[9]:


sentences_df = data.text.apply(lambda x: split_sentences(x))


# In[10]:


sep = int(len(sentences_df) * 0.8)


# In[11]:


train_sentences_txt = 'data/clear_data/train_sentences.txt'
heldout_sentences_txt = 'data/clear_data/heldout _sentences.txt'

with open(train_sentences_txt,'a') as f:
    for sentence_list in sentences_df[:sep]:
        for sentence in sentence_list:
            f.write(sentence[0].lower() + sentence[1:] + '\n')
        
with open(heldout_sentences_txt,'a') as f:
    for sentence_list in sentences_df[sep:]:
        for sentence in sentence_list:
            f.write(sentence[0].lower() + sentence[1:] + '\n')

