#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import razdel
from collections import Counter
from nltk.tokenize import sent_tokenize
from gensim.summarization.textcleaner import split_sentences
import re


# In[2]:


data = (pd.read_csv('data/avito/train.csv', usecols=['description', 'category_id'], nrows=25_000)
      .rename(columns={'description': 'text', 'category_id': 'label'})
     )


# In[3]:


import re

znaks = r"[,:!()?]"
def tokenize(string):
    string = re.sub(r"[^A-Za-zА-Яа-я0-9()\-,!?\'\`’:]", " ", string)
    string = re.sub(r"’|`", "'", string)
    string = re.sub(znaks, lambda x: f' {x.group()} ', string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    tokens = [token.text for token in razdel.tokenize(string)]
    return tokens
    
data['text'] = data['text'].apply(tokenize)


# In[4]:


fastText_txt = 'data/fast_emb/fastText.txt'

with open(fastText_txt, 'w') as f:
    for tok_text in data['text'].tolist():
        f.write(' '.join(tok_text)+'\n')


# In[2]:


get_ipython().system('pip3 install fasttext')


# In[1]:


import fasttext


# ## У них вроде все есть, но если что вот инструкция
# ###  В последней строке путь для ../fastText.txt  только какой нужно ставь
# 
# $ wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
# 
# $ unzip v0.1.0.zip
# 
# $ cd fastText-0.1.0
# 
# $ make
# 
# $ ./fasttext skipgram -input ../fastText.txt -output output -dim 1024 
