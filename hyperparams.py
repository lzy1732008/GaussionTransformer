# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
class Hyperparams:
    '''Hyperparameters'''
    is_training = True
    # data
    source_train = 'corpora/train.tags.de-en.de'
    target_train = 'corpora/train.tags.de-en.en'
    source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'
    char_vocab_size = 4594 # *
    word_vocab_size = 97505
    char_dimension = 30
    word_dimension = 128
    postion_dimension = 120

    
    # training
    batch_size = 64 # alias = N
    lr = 4e-5 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    num_epochs = 20
    save_per_batch = 100
    print_per_batch = 10


    # model
    # input param
    X_maxlen = 100 # Maximum number of words in a sentence. alias = T. *
    Y_maxLen = 100 #*

    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.

    #model param
    num_heads = 4
    dropout_rate = 0.1
    encoder_num_blocks = 3 # number of encoder/decoder blocks
    inter_num_blocks = 2


    
    
    
    
