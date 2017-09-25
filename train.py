#! /usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from CNN_network import My_CNN
from data_helper import DataHelper
from tensorflow.contrib import learn
import gzip
from random import random
import math

# Parameters
# ==================================================

tf.flags.DEFINE_integer("image_length", 48, "the length of image ")
tf.flags.DEFINE_integer("image_width", 48, "the width of image ")
tf.flags.DEFINE_integer("image_channels", 3, "the hannels of image ")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_classes", 43, "numble of classes")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

ROOT_PATH = "/data/MyTensorflow/MyProject/traffic-signs-master"
train_data_dir = os.path.join(ROOT_PATH, "traffic-signs-data/train/Final_Training/Images")
datahelper = DataHelper()
print('load data start ...')
train_set, dev_set, sum_no_of_batches = datahelper.getDataSets(train_data_dir, 10, FLAGS.batch_size, FLAGS.image_length, FLAGS.image_width, FLAGS.num_classes)
print('load data end ...')

# 从scores中取出前五 get label using probs
def get_label_using_probs(scores, top_number=5):
    index_list = np.argsort(scores)[-top_number:]
    index_list = index_list[::-1]
    return index_list

# 计算f1的值
def f1_eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = (precision * recall) / (precision + recall)

    return f1

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        Model = My_CNN(
            num_class=FLAGS.num_classes,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            batch_size=FLAGS.batch_size)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized siameseModel object")

    grads_and_vars = optimizer.compute_gradients(Model.cost)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", Model.cost)
    # acc_summary = tf.summary.scalar("accuracy", Model.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary,  grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

    # Write vocabulary
    # vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    def train_step(x1_batch, y_batch):
        """
        A single training step
        """

        feed_dict = {
            Model.input_x: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: FLAGS.dropout_keep_prob,
            Model.b_size: len(y_batch)
        }

        _, step, loss, accuracy = sess.run(
            [tr_op_set, global_step, Model.cost, Model.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        summary_op_out = sess.run(train_summary_op, feed_dict=feed_dict)
        train_summary_writer.add_summary(summary_op_out, step)


    def dev_step(x1_batch, y_batch):
        """
        A single dev step
        """

        feed_dict = {
            Model.input_x: x1_batch,
            Model.input_y: y_batch,
            Model.dropout_keep_prob: 1.0,
            Model.b_size: len(y_batch)
        }

        step, loss, accuracy = sess.run(
            [global_step, Model.cost, Model.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        summary_op_out = sess.run(train_summary_op, feed_dict=feed_dict)
        train_summary_writer.add_summary(summary_op_out, step)
        return accuracy


    # Generate batches
    batches = datahelper.batch_iter(
        list(zip(train_set[0], train_set[1])), FLAGS.batch_size, FLAGS.num_epochs)

    ptr = 0
    max_validation_acc = 0.0
    for nn in range(sum_no_of_batches * FLAGS.num_epochs):
        batch = batches.__next__()
        if len(batch) < 1:
            continue
        x1_batch, y_batch = zip(*batch)
        if len(y_batch) < 1:
            continue
        train_step(x1_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_batches = datahelper.batch_iter(list(zip(dev_set[0], dev_set[1])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db) < 1:
                    continue
                x1_dev_b, y_dev_b = zip(*db)
                if len(y_dev_b) < 1:
                    continue
                accuracy = dev_step(x1_dev_b, y_dev_b)
                sum_acc = sum_acc + accuracy
            print('-----the current accuracy is %f ----'%(sum_acc / (int(len(dev_set[0]) / FLAGS.batch_size) + 1)))
        if current_step % FLAGS.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(nn) + ".pb",
                                     as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc / (int(len(dev_set[0]) / FLAGS.batch_size) + 1),
                                                                                      checkpoint_prefix))
