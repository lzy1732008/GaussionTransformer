import hyperparams as param
import modules
import tensorflow as tf

class GaussionTransformer:
    def __init__(self):
        self.inputX_word = tf.placeholder(name="inputX_word", dtype=tf.float64,
                                          shape=[None, param.Hyperparams.X_maxlen, param.Hyperparams.word_dimension])
        self.inputX_char = tf.placeholder(name="inputX_char", dtype=tf.float64,
                                          shape=[None, param.Hyperparams.X_maxlen, param.Hyperparams.char_dimension])
        self.inputY_word = tf.placeholder(name="inputY_word", dtype=tf.float64,
                                          shape=[None, param.Hyperparams.X_maxlen, param.Hyperparams.word_dimension])
        self.inputY_char = tf.placeholder(name="inputY_char", dtype=tf.float64,
                                          shape=[None, param.Hyperparams.X_maxlen, param.Hyperparams.char_dimension])
        self.y = tf.placeholder(name="target_y", dtype=tf.int32, shape=[None, 2])

        self.dropout_rate = tf.placeholder(tf.float64, name='keep_prob')

        self.run_GaussionTransformer()

    def run_GaussionTransformer(self):
        embeddingScope = "embeddingBlock"
        encodingBlock = "encodingBlcok"
        interactionBlock = "interactionBlock"
        comparisonBlock = "comparisonBlock"

        self.positionEncoding1 = modules.positional_encoding(inputs=self.inputX_word,
                                                             num_units=param.Hyperparams.postion_dimension) #(N, L, postion_dimension)
        self.positionEncoding2 = modules.positional_encoding(inputs=self.inputY_word,
                                                             num_units=param.Hyperparams.postion_dimension)
        self.shift = tf.Variable(tf.random_normal([1],stddev=0, seed=0,dtype=tf.float64) ** 2 + 0.001, trainable=True, name='shift', dtype=tf.float64)
        self.bias = tf.Variable(-tf.random_normal([1],stddev=0, seed=0,dtype=tf.float64) ** 2, trainable=True, name='bias', dtype=tf.float64)


        with tf.variable_scope(embeddingScope,reuse=False):
            self.embedding_1 = modules.embedding_block(self.inputX_word,
                                                       self.inputX_char,
                                                       self.positionEncoding1,
                                                       scope="embedding_1")

            self.embedding_2 = modules.embedding_block(self.inputY_word,
                                                       self.inputY_char,
                                                       self.positionEncoding2,
                                                       scope="embedding_2")

        with tf.variable_scope(encodingBlock, reuse=False):
            self.encoding_1 = self.embedding_1
            self.encoding_2 = self.embedding_2
            for i in range(param.Hyperparams.encoder_num_blocks):
                with tf.variable_scope("multihead-atttention_{0}".format(i), reuse=False):#这里添加scope， {}.format
                    self.encoding_1 = modules.multihead_attention(self.encoding_1,
                                                                  self.shift,
                                                                  self.bias,
                                                                  num_heads=param.Hyperparams.num_heads,
                                                                  dropout_rate=self.dropout_rate,
                                                                  is_training=param.Hyperparams.is_training)

                with tf.variable_scope("multihead-atttention_{0}".format(i), reuse=True):  # 这里添加scope， {}.format
                    self.encoding_2 = modules.multihead_attention(self.encoding_2,
                                                                  self.shift,
                                                                  self.bias,
                                                                  num_heads=param.Hyperparams.num_heads,
                                                                  dropout_rate=self.dropout_rate,
                                                                  is_training=param.Hyperparams.is_training)



            self.encoding_1 += self.positionEncoding1
            self.encoding_2 += self.positionEncoding2

        with tf.variable_scope(interactionBlock, reuse=None):
            self.interaction_1 = self.encoding_1
            self.interaction_2 = self.encoding_2

            for i in range(param.Hyperparams.inter_num_blocks):
                with tf.variable_scope("interaction_{0}".format(i), reuse=False):
                    self.interaction_1 = modules.InteractionBlock(queries=self.interaction_1,
                                                                  keys=self.interaction_2,
                                                                  shift=self.shift,
                                                                  bias=self.bias,
                                                                  num_heads=param.Hyperparams.num_heads,
                                                                  dropout_rate=self.dropout_rate,
                                                                  is_training=param.Hyperparams.is_training)
                with tf.variable_scope("interaction_{0}".format(i), reuse=True):
                    self.interaction_2 = modules.InteractionBlock(queries=self.interaction_2,
                                                                  keys=self.interaction_1,
                                                                  shift=self.shift,
                                                                  bias=self.bias,
                                                                  num_heads=param.Hyperparams.num_heads,
                                                                  dropout_rate=self.dropout_rate,
                                                                  is_training=param.Hyperparams.is_training)


        with tf.variable_scope(comparisonBlock, reuse=None):
            self.logit = modules.ComparisonBlock(input1_Encoding=self.encoding_1,
                                             input1_Interaction=self.interaction_1,
                                             input2_Encoding=self.encoding_2,
                                             input2_Interaction=self.interaction_2)
            self.pred_y = tf.argmax(self.logit,1)
            if param.Hyperparams.is_training:
                with tf.name_scope("optimize"):
                    # 损失函数，交叉熵
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logit,
                                                                            labels=self.y)  # 对logits进行softmax操作后，做交叉墒，输出的是一个向量
                    # regularizer = tf.contrib.layers.l2_regularizer(scale=5.0 / 50000)
                    # reg_term = tf.contrib.layers.apply_regularization(regularizer)
                    self.loss = tf.reduce_mean(cross_entropy)  # 将交叉熵向量求和，即可得到交叉熵
                    # 优化器
                    self.optim = tf.train.AdamOptimizer(learning_rate=param.Hyperparams.lr).minimize(self.loss)

                with tf.name_scope("accuracy"):
                    # 准确率
                    correct_pred = tf.equal(tf.argmax(self.y, 1),
                                            self.pred_y)  # 由于input_y也是onehot编码，因此，调用tf.argmax(self.input_y)得到的是1所在的下表
                    self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
