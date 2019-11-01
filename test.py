import tensorflow as tf
import numpy as np

# lookup_table = np.reshape(range(16),newshape=[4,4])
# inputs = tf.convert_to_tensor(np.array([[[0,1],[0,2]],[[1,3],[2,3]]]))
# # output = tf.nn.embedding_lookup(lookup_table, inputs)
# position_ind = tf.tile(tf.expand_dims(tf.range(10), 0), [100, 1])
# T_q = 3
# T_k = 4
# w = 0.1
# b = -0.2
# Nq = 4
#
# Dis_M = np.zeros(shape=[T_q, T_k])
# for i in range(T_q):
#     for j in range(T_k):
#         Dis_M[i][j] = -abs(w * (i - j) ** 2 + b)
# dis_M = tf.convert_to_tensor(Dis_M)
# dis_M = tf.concat(tf.tile(tf.expand_dims(dis_M, 0), [Nq, 1, 1]),axis=0)
#
# dis_split = tf.split(dis_M, Nq)
#
#
# with tf.Session() as sess:
#     with tf.variable_scope("interaction-1", reuse=False):
#         with tf.variable_scope("sub-layer"):
#             shift = tf.convert_to_tensor(8,name="shift")
#
#     with tf.variable_scope("interaction 1", reuse=True):
#         with tf.variable_scope("sub-layer"):
#             shift1 = tf.get_variable("shift")
#             assert shift == shift1


import Model
inputX_word = np.random.rand(64,100,300)
inputX_char = np.random.rand(64,100,30)
inputY_word = np.random.rand(64,100,300)
inputY_char = np.random.rand(64,100,30)
input_y = [[0, 1] for _ in range(64)]

model = Model.GaussionTransformer()
feed_dict = {
        model.inputX_word: inputX_word,
        model.inputX_char: inputX_char,
        model.inputY_word: inputY_word,
        model.inputY_char: inputY_char,
        model.y: input_y
    }


with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)
    y = sess.run(model.pred_y, feed_dict=feed_dict)

    print(y)






