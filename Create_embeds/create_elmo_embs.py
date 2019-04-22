#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bilm razdel')


# In[ ]:


from bilm.training import dump_weights as dw

dw('../input/model-elmo/', 'weights.hdf5')


# In[ ]:


import pandas as pd

df = (pd.read_csv('../input/avito-dataset/train.csv', usecols=['description', 'category_id'], nrows=25_000)
      .rename(columns={'description': 'text', 'category_id': 'label'}))


# In[ ]:


import json

with open('../input/model-elmo/options.json', 'r') as file:
    options = json.load(file)
    
options['char_cnn']['n_characters'] += 1
    
with open('options.json', 'w') as file:
    json.dump(options, file)


# In[ ]:


import re
import razdel

znaks = r"[,:!()?]"
def tokenize(string):
    string = re.sub(r"[^A-Za-zА-Яа-я0-9()\-,!?\'\`’:]", " ", string)
    string = re.sub(r"’|`", "'", string)
    string = re.sub(znaks, lambda x: f' {x.group()} ', string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()
    tokens = [token.text for token in razdel.tokenize(string)]
    return tokens


# In[ ]:


from bilm import dump_bilm_embeddings
import os
import h5py

df = df.text.apply(tokenize).tolist()

dataset_file = 'dataset_file.txt'
with open(dataset_file, 'w') as fout:
    for sentence in df:
        fout.write(' '.join(sentence) + '\n')


# In[ ]:


vocab_file = os.path.join('../input/clear-data-elmo', 'vocab.txt')
options_file = os.path.join('.', 'options.json')
weight_file = 'weights.hdf5'
embedding_file = 'elmo_embeddings.hdf5'


# In[ ]:


get_ipython().run_cell_magic('time', '', 'dump_bilm_embeddings(vocab_file, dataset_file,\n                     options_file, weight_file,\n                     embedding_file)')

