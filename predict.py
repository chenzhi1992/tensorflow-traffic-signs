#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys
import os.path
import time
import datetime
from sys import argv
from data_helper import DataHelper


class Traffic_sign_cnn(object):
    def __init__(self, meta_path, model_path):
        print(" meta dir: " + meta_path)
        print(" model dir: " + model_path)

        tf.flags.DEFINE_integer("batch_size_classify", 1, "Batch Size (default: 64)")
        tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
        tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

        self.FLAGS = tf.flags.FLAGS
        self.FLAGS._parse_flags()

        # Evaluation
        # ==================================================
        # checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            with tf.device("/gpu:0"):
                session_conf = tf.ConfigProto(
                    allow_soft_placement=self.FLAGS.allow_soft_placement,
                    log_device_placement=self.FLAGS.log_device_placement)
                session_conf.gpu_options.allow_growth = True
                self.sess = tf.Session(config=session_conf)
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph(meta_path)
                saver.restore(self.sess, model_path)

                # for op in graph.get_operations():
                #     print(op.name)
                # Get the placeholders from the graph by name
                self.input_x = graph.get_tensor_by_name("input_x:0")
                # self.input_y = graph.get_tensor_by_name("input_y:0")
                self.dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
                self.b_size = graph.get_tensor_by_name("batch_size:0")

                # Tensors we want to evaluate
                self.predict = graph.get_tensor_by_name("accuracy/prediction:0")

    def __enter__(self):
        print('Sentence_Similarity enter')

    def __exit__(self):
        print('Sentence_Similarity exit')
        self.sess.close()

    def getlabel(self, image):

        feed_dict = {
            self.input_x: image,
            self.dropout_keep_prob: 1.0,
            self.b_size: self.FLAGS.batch_size_classify
        }
        label = self.sess.run(self.predict, feed_dict)
        return label


if __name__ == '__main__':
    cnn_meta_path = './runs/model2/model-280000.meta'
    cnn_model_path = './runs/model2/model-280000'
    ts_cnn = Traffic_sign_cnn(cnn_meta_path, cnn_model_path)

    ROOT_PATH = "/data/MyTensorflow/MyProject/traffic-signs-master"
    test_data_dir = os.path.join(ROOT_PATH, "traffic-signs-data/test/Final_Test/Images")
    test_label_file = os.path.join(ROOT_PATH, "traffic-signs-data/test/GT-final_test.csv")
    datahelper = DataHelper()
    images_test, labels_test = datahelper.load_test_data(test_data_dir, test_label_file)

    predict_rigth = 0
    start = time.time()
    for i in range(len(images_test)):
        image = np.reshape(images_test[i], (1, 48, 48, 1))
        label = ts_cnn.getlabel(image)
        if label[0] == int(labels_test[i]):
            predict_rigth += 1
        else:
            print('第张图片不对，实际是，预测是', (i, int(labels_test[i]), label[0]))
    end = time.time()
    print('avg time test an image is: %f' % ((end - start) / len(images_test)) )
    print('test acc is ', (predict_rigth / len(images_test)))

    # =========================================================================================================
