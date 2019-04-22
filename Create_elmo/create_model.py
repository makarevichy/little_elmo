#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bilm')


# In[ ]:


get_ipython().system('export CUDA_VISIBLE_DEVICES=0,1,2')


# In[ ]:


import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(train_prefix, vocab_file, save_dir):
    # load the vocab
    vocab = load_vocab(vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 3

    # number of tokens in training data (this for 1B Word Benchmark)
    #n_train_tokens = 768_648_884
    n_train_tokens = 1_246_091

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],],
       #[4, 128],
       #[5, 256],
       #[6, 512],
       #[7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      #'projection_dim': 512,
       'projection_dim': 64,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 2,#10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = save_dir
    tf_log_dir = save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


# In[ ]:


main(train_prefix='../input/train_sentences.txt', vocab_file='../input/vocab.txt', save_dir='.')

