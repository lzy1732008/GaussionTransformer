# coding: utf-8
#包含5个输入关键参数：inputx,inputx_len,inputy,inputy_len,y
from __future__ import print_function


import sys
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics
import os
from Model import GaussionTransformer
from data_load import *
import hyperparams as hp

save_dir = 'result/model/GaussionTransformer'  #修改处
param_des = 'encoder-2@inter-3@dp-0.5'
save_path = os.path.join(save_dir,param_des+'/checkpoints/best_validation')
tensorboard_dir = os.path.join(save_dir,param_des+'/tensorboard')

model = GaussionTransformer()

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(a_word,a_char,b_word,b_char,y_batch,dropout_rate):
    feed_dict = {
        model.inputX_char: a_char,
        model.inputX_word: a_word,
        model.inputY_char: b_char,
        model.inputY_word: b_word,
        model.y: y_batch,
        model.dropout_rate: dropout_rate,
    }

    return feed_dict


def evaluate(sess,a_word,a_char,b_word,b_char,y):
    """评估在某一数据上的准确率和损失"""
    data_len = len(a_word)
    batch_eval = get_batch_data_test(a_word, a_char, b_word, b_char,y, hp.Hyperparams.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for a_word_batch, a_char_batch, b_word_batch, b_char_batch,y_batch in batch_eval:
        batch_len = len(a_char_batch)
        feed_dict = feed_data(a_word_batch, a_char_batch, b_word_batch, b_char_batch,y_batch,1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖

    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    #结果可视化与存储
    tf.summary.scalar("loss", model.loss) #可视化loss
    tf.summary.scalar("accuracy", model.acc)  #可视化acc
    merged_summary = tf.summary.merge_all()   #将所有操作合并输出
    writer = tf.summary.FileWriter(tensorboard_dir) #将summary data写入磁盘

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    train_data, test_data, val_data = data_load()

    train_x1_word, train_x1_char, train_x2_word, train_x2_char, train_y = train_data
    val_x1_word, val_x1_char, val_x2_word, val_x2_char, val_y = val_data

    print('train len',len(train_x1_char))
    print('val_len',len(val_x1_char))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(hp.Hyperparams.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = get_batch_data(train_x1_word, train_x1_char, train_x2_word, train_x2_char, train_y, hp.Hyperparams.batch_size)
        for a_word_batch, a_char_batch, b_word_batch, b_char_batch,y_batch in batch_train:
            feed_dict = feed_data(a_word_batch, a_char_batch, b_word_batch, b_char_batch,y_batch,hp.Hyperparams.dropout_rate)

            if total_batch % hp.Hyperparams.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % hp.Hyperparams.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能

                feed_dict[model.dropout_rate] = 1.0
                loss_train, acc_train,pre_y, logit, true_y, encodings = session.run([model.loss, model.acc,model.pred_y,model.logit,model.y,model.encoding_1], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, val_x1_word, val_x1_char, val_x2_word, val_x2_char, val_y)  # 验证当前会话中的模型的loss和acc
                # for pre_y_, logit_, true_y_ in zip(pre_y,logit,true_y):
                #     print(pre_y_, logit_, true_y_)


                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0}, Train Loss: {1}, Train Acc: {2},' \
                      + ' Val Loss: {3}, Val Acc: {4}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break

def test():
    print("Loading test data...")
    start_time = time.time()
    test_data = data_load_test()
    test_x1_word, test_x1_char, test_x2_word, test_x2_char, test_y = test_data


    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, test_x1_word, test_x1_char, test_x2_word, test_x2_char, test_y)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = hp.Hyperparams.batch_size
    data_len = len(test_x1_char)
    num_batch = int((data_len) / batch_size)

    y_test_cls = np.argmax(test_y, 1)
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.inputX_word: test_x1_word[start_id:end_id],
            model.inputX_char: test_x1_char[start_id:end_id],
            model.inputY_word: test_x2_word[start_id:end_id],
            model.inputY_char: test_x2_char[start_id:end_id],
            model.y: test_y,
            model.dropout_rate: 1.0   #这个表示测试时不使用dropout对神经元过滤
        }
        y_pred_cls[start_id:end_id] = session.run(model.pred_y, feed_dict=feed_dict)   #将所有批次的预测结果都存放在y_pred_cls中



    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls,digits=4))#直接计算准确率，召回率和f值

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return y_test_cls,y_pred_cls

train()
# y_test_cls,y_pred_cls = test()
# wsnamels = getwslist(model=model)
# wsevaluate(y_test_cls, y_pred_cls,wsnamels)