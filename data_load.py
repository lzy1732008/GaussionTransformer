# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
import numpy as np
import msgpack
import preprocess as prep
import tensorflow.contrib.keras as kr
import hyperparams as hp
import json

def get_batch_data_test(a_word,a_char,b_word,b_char,y, batch_size = 64):
    data_len = len(a_word)
    num_batch = int(data_len/batch_size)

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield a_word[start_id:end_id],a_char[start_id:end_id],\
              b_word[start_id:end_id],b_char[start_id:end_id],\
              y[start_id:end_id]


def get_batch_data(a_word,a_char,b_word,b_char,y,batch_size = 64):
    data_len = len(a_word)
    num_batch = int(data_len/batch_size)

    indices = np.random.permutation(np.arange(data_len))
    a_word_shuffle = a_word[indices]
    a_char_shuffle = a_char[indices]
    b_word_shuffle = b_word[indices]
    b_char_shuffle = b_char[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = batch_size * i
        end_id = min(batch_size * (i + 1), data_len)
        yield a_word_shuffle[start_id:end_id],a_char_shuffle[start_id:end_id],\
              b_word_shuffle[start_id:end_id],b_char_shuffle[start_id:end_id],\
              y_shuffle[start_id:end_id]


def data_load(envPath):
    with open(envPath,'r',encoding='utf-8') as fr:
        env = json.load(fr)

    train_data = env['train']
    test_data = env['test']
    val_data = env['val']
    train = processInitData(train_data)
    test = processInitData(test_data)
    val = processInitData(val_data)
    return train,test,val

#process data from env
def processInitData(data):
    a_data_word = []
    a_data_char = []
    b_data_word = []
    b_data_char = []
    y = []

    for sample in data:
        assert len(sample) == 3, ValueError("the number of elemengs in this sample is {0}".format(len(sample)))
        input_a, input_b, target_y = sample[0],sample[1],int(sample[2])
        a_words, a_chars = input_a['word_input'], input_a['char_input']
        b_words, b_chars = input_b['word_input'], input_b['char_input']
        a_data_word.append(list(map(lambda x:prep.getVector(x), a_words)))
        a_data_char.append(list(map(lambda x:prep.getVector(x), a_chars)))
        b_data_word.append(list(map(lambda x:prep.getVector(x), b_words)))
        b_data_char.append(list(map(lambda x:prep.getVector(x), b_chars)))
        if target_y == 1:
            y.append([0,1])
        else:
            y.append([1,0])

    a_data_char = kr.preprocessing.sequence.pad_sequences(np.array(a_data_char), hp.Hyperparams.X_maxlen)
    a_data_word = kr.preprocessing.sequence.pad_sequences(np.array(a_data_word), hp.Hyperparams.X_maxlen)
    b_data_char = kr.preprocessing.sequence.pad_sequences(np.array(b_data_char), hp.Hyperparams.Y_maxLen)
    b_data_word = kr.preprocessing.sequence.pad_sequences(np.array(b_data_word), hp.Hyperparams.Y_maxLen)
    return a_data_char,a_data_word,b_data_char,b_data_word, y



# def getVector(vectorStr):
#     vectorStr = vectorStr[1:-1]
#     vectorStr = str(vectorStr).replace(',','')
#     vectorStr = str(vectorStr).replace('\'','')
#     vectors = vectorStr.split('\t')
#     vectors = list(map(float, map(lambda x: x.strip(), filter(lambda x: x.strip() != '', vectors))))
#     assert len(vectors) == hp.Hyperparams.char_dimension or len(vectors) == hp.Hyperparams.word_dimension, ValueError("wrong dimension:{0}".format(len(vectors)))
#     return vectors


