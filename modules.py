# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import hyperparams as param

#Batch Normalization
def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape,dtype=tf.float64),dtype=tf.float64)
        gamma = tf.Variable(tf.ones(params_shape,dtype=tf.float64),dtype=tf.float64)
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
        
    return outputs

#其实就是随机初始化一个矩阵M（vocab_size, embedding_size）,然后拿输入input乘上这个矩阵
def wordEmbedding(inputs,
              vocab_size,
              num_units, 
              zero_pad=True, 
              scale=True,
              scope="embedding", 
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
        
    For example,
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```
    
    ```
    import tensorflow as tf
    
    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        
        if scale:
            outputs = outputs * (num_units ** 0.5) 
            
    return outputs


def charEmbedding(inputs,
                  max_len,
                  inputs_word_lens,
                  char_vocab_size,
                  num_units,
                  scope="charEmbedding",
                  reuse=None):
    ''' embedding char
    :param inputs: a 3d tensor with shape [N, T_len, T_wordLen]
    :param num_units: the dimension of char embedding
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name
    :return: char embedding with shape [N, T_len, T_char_embedding].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table_char',
                                       dtype=tf.float32,
                                       shape=[char_vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        inputs_flatten = tf.layers.Flatten(inputs, name=scope + 'flatten_char')
        outputs = tf.nn.embedding_lookup(lookup_table,inputs_flatten)
        batch_size = inputs.get_shape()[0]
        new_outputs = np.zeros(shape=[batch_size,max_len,num_units])

        for i in range(batch_size):
            base_index = 0
            for index, j in enumerate(inputs_word_lens[i]):
                if j == -1:
                    break
                new_outputs[i][index] = tf.reduce_max(outputs[i][base_index:base_index + j], axis=1)
                base_index += j

        new_outputs = tf.convert_to_tensor(new_outputs)
        return new_outputs

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: the inputs (N,L,dimension)
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()[:-1]
    if N is None:
        N = 64
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units],dtype=tf.float64),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs

def embedding_block(input_words,
                    input_char,
                    position_encoding,
                    num_units=None,
                    scope="single_embeddingblock",
                    reuse=None):
    '''
    a Embedding Block
    :param input_words: input with word vector [N, L, wordVector]
    :param input_char:  input with char vector [N, L, charVector]
    :param num_units:  the output dimension of feedforward
    :param position_encoding: postinal encoding [N, L , num_units]
    :param scope:
    :param reuse:
    :return: positional encoding + [word; char] * W
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = position_encoding.get_shape().as_list()[-1]
        inputs = tf.concat([input_words, input_char], axis=2)
        outputs = position_encoding + tf.layers.dense(inputs=inputs, units=num_units, name="feedforward")
        return outputs


def Gaussion_selfAttention(queries,
                           keys,
                           shift,
                           bias,
                           num_units,
                           scope="Gaussion_selfAttention",
                           num_heads=8,
                           dropout_rate=0,
                           is_training=True,
                           causality=False,
                           reuse=None,
                           if_Gausssion=True):
    with tf.variable_scope(scope, reuse=reuse):

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        Nq, T_q, Cq = Q.get_shape().as_list()
        Nk, T_k, Ck = K.get_shape().as_list()
        assert Nq == Nk, ValueError(
            "The number of queries is not equal to that of keys, they are {0}, and {1}".format(Nq, Nk))
        # Scale + Gaussion prior
        if if_Gausssion:
            Dis_M = np.zeros(shape=[T_q, T_k])
            for i in range(T_q):
                for j in range(T_k):
                    Dis_M[i][j] = (i - j) ** 2
            dis_M = tf.convert_to_tensor(Dis_M,dtype=tf.float64)

            shift_M = tf.tile(tf.tile(tf.expand_dims(shift,0), [T_q, 1]),[1, T_k])
            bias_M = tf.tile(tf.tile(tf.expand_dims(bias, 0), [T_q, 1]),[1, T_k])

            dis_M = -(shift_M * dis_M + bias_M)
            dis_M_ = tf.tile(tf.expand_dims(dis_M, 0), [Nq * num_heads, 1, 1])  # (h * N, T_q, T_k)

            outputs = (dis_M_ + outputs) / (K_.get_shape().as_list()[-1] ** 0.5)  # (h * N, T_q, T_k)
        else:
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # # Key Masking
        # key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        # key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        # key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        #
        # paddings = tf.ones_like(outputs)*(-2**32+1)
        # outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # # Query Masking
        # query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        # query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        # query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        # outputs *= query_masks # broadcasting. (N, T_q, C)
        #
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # add and layer norm
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (h, T_q, C)
        outputs += queries
        outputs = normalize(outputs)  # (N, T_q, C)
    return outputs

def multihead_attention(inputs,
                        shift,
                        bias,
                        num_units_attention=None,
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    # Set the fall back option for num_units

    with tf.variable_scope(scope, reuse=reuse):
        if num_units_attention is None:
            num_units = inputs.get_shape().as_list()[-1]
        else:
            num_units = num_units_attention

        outputs = Gaussion_selfAttention(queries=inputs,
                                         keys=inputs,
                                         shift=shift,
                                         bias=bias,
                                         num_units=num_units,
                                         num_heads=num_heads,
                                         dropout_rate=dropout_rate,
                                         is_training=is_training)
        # feedforward
        outputs_ff = tf.nn.relu(tf.layers.dense(inputs=outputs,units=num_units,name="ffn_1"))
        outputs_ff = tf.layers.dense(inputs=outputs_ff, units=num_units, name="ffn_2")

        # add and Layer norm
        outputs += outputs_ff #(N, T_q, C)
        outputs = normalize(outputs) #(N, T_q, C)

        return outputs

def InteractionBlock(queries,
                     keys,
                     shift,
                     bias,
                     num_units_attention=None,
                     num_heads=8,
                     dropout_rate=0,
                     is_training=True,
                     causality=False,
                     scope="interaction_block",
                     reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
         # Gaussion + self-attention
         if num_units_attention is None:
             num_units = queries.get_shape().as_list()[-1]
         else:
             num_units = num_units_attention

         Gaussion_output = Gaussion_selfAttention(queries=queries,
                                                  keys=queries,
                                                  shift=shift,
                                                  bias=bias,
                                                  num_units=num_units,
                                                  scope="self-attention",
                                                  num_heads=num_heads,
                                                  dropout_rate=dropout_rate,
                                                  is_training=is_training)

         # add and Norm
         Gaussion_output += queries
         Gaussion_output  = normalize(Gaussion_output)

         # interaction
         Interaction_output = Gaussion_selfAttention(queries=queries,
                                                     keys=keys,
                                                     shift=0,
                                                     bias=0,
                                                     num_units=num_units,
                                                     scope="interaction",
                                                     num_heads=num_heads,
                                                     dropout_rate=dropout_rate,
                                                     is_training=is_training,
                                                     if_Gausssion=False)

         # add and Norm
         Interaction_output += Gaussion_output

         # feedforward
         outputs_ff = tf.nn.relu(tf.layers.dense(inputs=Interaction_output, units=num_units, name="ffn_1"))
         outputs_ff = tf.layers.dense(inputs=outputs_ff, units=num_units, name="ffn_2")

         # add and Layer norm
         Interaction_output += outputs_ff  # (N, T_q, C)
         outputs = normalize(Interaction_output)  # (N, T_q, C)
    return outputs

def ComparisonBlock(input1_Interaction,
                    input1_Encoding,
                    input2_Interaction,
                    input2_Encoding,
                    scope="comparison_blcok",
                    reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        output1 = preComparison(input1_Interaction,
                                input1_Encoding,
                                scope="preComparison_1")
        output2 = preComparison(input2_Interaction,
                                input2_Encoding,
                                scope="preComparison_2")
        N1, T1, dimension1 = input1_Encoding.get_shape().as_list()
        N2, T2, dimension2 = input2_Encoding.get_shape().as_list()
        assert N1 == N2, ValueError(
            "Two input have different batch size, they are {0}, {1}".format(N1, N2))
        assert dimension1 == dimension2, ValueError(
            "Two input have different dimension, they are {0}, {1}".format(dimension1, dimension2))
        assert T1 == T2, ValueError(
            "Two input have different Length, they are {0}, {1}".format(T1, T2))

        concat_output = tf.concat([output1,output2],axis=-1)
        output_ff = tf.nn.relu(tf.layers.dense(inputs=concat_output,units=dimension1,name="predict_dense_1"))
        output = tf.nn.softmax(tf.layers.dense(inputs=output_ff,units=2,name="predict_dense_2"))
    return output

def preComparison(input_interaction,
                  input_encoding,
                  scope="preComparison",
                  reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        concat_input = tf.concat([input_interaction, input_encoding], axis=-1)
        N1, T1, dimension1 = input_encoding.get_shape().as_list()
        N2, T2, dimension2 = input_interaction.get_shape().as_list()
        assert N1 == N2, ValueError(
            "Two input have different batch size, they are {0}, {1}".format(N1, N2))
        assert dimension1 == dimension2, ValueError(
            "Two input have different dimension, they are {0}, {1}".format(dimension1, dimension2))
        assert T1 == T2, ValueError(
            "Two input have different Length, they are {0}, {1}".format(T1, T2))
        # mlp
        output_ff = tf.nn.relu(tf.layers.dense(inputs=concat_input,
                                               units=dimension1,
                                               name="feedback_1"))
        output_ff = tf.layers.dense(inputs=output_ff,
                                               units=dimension1,
                                               name="feedback_2")
        # scale
        reduce_output = tf.reduce_sum(output_ff, axis=1)
        output = reduce_output * 1.0 / (T1 ** (0.5))
    return output


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.
    
    For example,
    
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)
       
    outputs = label_smoothing(inputs)
    
    with tf.Session() as sess:
        print(sess.run([outputs]))
    
    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


            
