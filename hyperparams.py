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
    trainPath = 'resource/gyshz/train-init.txt'
    valPath = 'resource/gyshz/val-init.txt'
    testPath = 'resource/gyshz/test-init.txt'
    char_vocab_size = 4594 # *
    word_vocab_size = 97505
    char_dimension = 30
    word_dimension = 128
    postion_dimension = 120

    
    # training
    batch_size = 64 # alias = N
    lr = 4e-5 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory
    num_epochs = 200
    save_per_batch = 100
    print_per_batch = 10


    # model
    # input param
    X_maxlen = 30 # Maximum number of words in a sentence. alias = T. *
    Y_maxLen = 30 #*

    min_cnt = 20 # words whose occurred less than min_cnt are encoded as <UNK>.

    #model param
    num_heads = 4
    dropout_rate = 0.5
    encoder_num_blocks = 2 # number of encoder/decoder blocks
    inter_num_blocks = 3



    
    
    
    
