#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


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


# In[ ]:


df = (pd.read_csv('../input/train.csv')
      .drop(['title', 'price'], axis=1)
      .set_index('item_id')
      .rename(columns={'description': 'text', 'category_id': 'label'})
     )


# In[ ]:


import re


ends = r"(\'s)|(\'ve)|(n\'t)|(\'re)|(\'ll)"
znaks = r"[,:!()?]"


def clear_text(string):
    string = re.sub(r"[^A-Za-zА-Яа-я0-9(),!?\'\`’:]", " ", string)
    string = re.sub(r"’|`", "'", string)
    string = re.sub(ends, lambda x: f' {x.group()}', string)
    string = re.sub(znaks, lambda x: f' {x.group()} ', string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
    
df['text'] = df['text'].apply(clear_text)


# In[ ]:


len_text = 50
vocab_size = 5000
n_classes = len(df['label'].unique())


# In[ ]:


tok = Tokenizer(num_words=vocab_size)
tok.fit_on_texts(df['text'].tolist()) # corpus


# In[ ]:


def vectorize(tokenizer, data, maxlen):
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, maxlen=maxlen, padding='post')
    return data

X = vectorize(tok, df['text'].values, len_text)


# In[ ]:


y = np.zeros((len(df), n_classes))
for i, val in enumerate(df['label'].values):
    y[i, val] = 1


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
BS = 200
emb_size = 150
kernel_size = 10
emb_len = len(X[0])


# In[ ]:


layers = [
          Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=emb_len),
          Conv1D(filters=emb_size, kernel_size=kernel_size, padding="same", activation="relu"),
          MaxPooling1D(),
          Flatten(),
          Dropout(0.2),
          Dense(n_classes, activation="softmax")
]

model = Sequential()
for layer in layers:
    model.add(layer)


# In[ ]:


model.summary()


# In[ ]:


opt = SGD(lr=INIT_LR)
        
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)

model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BS)


# In[ ]:


acc = model.evaluate(X_eval, y_eval, batch_size=128)[1]


# In[ ]:


print(f'ACCURACY = {acc:.3f}')

