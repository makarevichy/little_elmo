#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install bilm')


# In[ ]:


get_ipython().system('export CUDA_VISIBLE_DEVICES=0')


# In[ ]:


from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset

def main(test_prefix, vocab_file, save_dir, batch_size=256):
    options, ckpt_file = load_options_latest_checkpoint(save_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(vocab_file, max_word_length)

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    else:
        data = LMDataset(test_prefix, vocab, **kwargs)

    test(options, ckpt_file, data, batch_size=batch_size)


# In[ ]:


main(test_prefix='../input/data-elmo/h*',
     vocab_file='../input/data-elmo/vocab.txt',
     save_dir='../input/model-elmo/')

