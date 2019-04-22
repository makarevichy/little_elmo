#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install razdel')


# In[ ]:


import pandas as pd
import numpy as np
import razdel
import h5py


# In[ ]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Embedding, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from gensim.models import KeyedVectors
from keras.layers import Input, Embedding
from keras.models import Model


# In[ ]:


df = (pd.read_csv('../input/avito-dataset/train.csv', usecols=['description', 'category_id'], nrows=25_000)
      .rename(columns={'description': 'text', 'category_id': 'label'})
     )


# In[ ]:


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
    
df['text'] = df['text'].apply(tokenize)


# In[ ]:


FT_model = KeyedVectors.load_word2vec_format('../input/fasttextemb/output.vec')

def extract_embs():
    embs = []
    for text in df['text']:
        embs.append(np.vstack([FT_model.word_vec(token) for token in text]))
    return embs
#text_vector = np.sum(tok_vec, axis=0)
#return text_vector / np.linalg.norm(text_vector)


# In[ ]:


def create_input(embs):
    return np.vstack([emb.mean(axis=0) for emb in embs])


# In[ ]:


n_classes = len(df['label'].unique())

y = np.zeros((len(df), n_classes))
for i, val in enumerate(df['label'].values):
    y[i, val] = 1


# In[ ]:


X = create_input(extract_embs())


# In[ ]:


ind_train, ind_eval = train_test_split(np.arange(len(df)), test_size=0.3, random_state=24)

X_train = X[ind_train]
X_eval = X[ind_eval]

y_train = y[ind_train]
y_eval = y[ind_eval]


# # Model

# In[ ]:


INIT_LR = 0.01
EPOCHS = 100
BS = 20


# In[ ]:


layers = [Dense(16, activation="relu"),
          Dense(128, activation="relu"),
          Dense(64, activation="tanh"),
          Dropout(0.2),
          Dense(n_classes, activation="softmax")
]

model = Sequential()
for layer in layers:
    model.add(layer)


# In[ ]:


#model.summary()


# In[ ]:


opt = SGD(lr=INIT_LR)

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)


# In[ ]:


model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BS)


# In[ ]:


acc = model.evaluate(X_eval, y_eval, batch_size=20)[1]


# In[ ]:


print(f'ACCURACY = {acc:.3f}')


# In[ ]:




