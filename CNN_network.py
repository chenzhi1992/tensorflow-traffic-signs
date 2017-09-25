import tensorflow as tf
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average

class My_CNN(object):

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        # stride [1, x_movement, y_movement, 1]
        # Must have strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pooling(self, x, size):
        # stride [1, x_movement, y_movement, 1]
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')

    def batch_norm(x,  train_phase, eps=1e-05, decay=0.9, affine=True, name=None):
        with tf.variable_scope(name, default_name='BatchNorm2d'):
            params_shape = tf.shape(x)[-1:]
            moving_mean = tf.get_variable('mean', params_shape,
                                          initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_variance = tf.get_variable('variance', params_shape,
                                              initializer=tf.ones_initializer,
                                              trainable=False)

            def mean_var_with_update():
                mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
                with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                              assign_moving_average(moving_variance, variance, decay)]):
                    return tf.identity(mean), tf.identity(variance)

            mean, variance = tf.cond(train_phase, mean_var_with_update, lambda: (moving_mean, moving_variance))
            if affine:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
            return x

    def __init__(self, num_class, l2_reg_lambda, batch_size):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, 48, 48, 1], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None, num_class], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.b_size = tf.placeholder(tf.int32, [], name='batch_size')#不固定batch_size

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        with tf.name_scope("cnn"):
            ## conv1 layer ##
            W_conv1 = self.weight_variable([5, 5, 1, 32])  # patch 5x5, in size 3, out size 32
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(self.input_x, W_conv1) + b_conv1)  # output size 32x32x32 (有padding的过程,下面一样)
            # conv2d -> batch_norm -> relu
            h_pool1 = self.max_pooling(h_conv1, 2)  # output size 16x16x32
            h_pool1 = tf.nn.dropout(h_pool1, self.dropout_keep_prob)

            ## conv2 layer ##
            W_conv2 = self.weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)  # output size 16x16x64
            h_pool2 = self.max_pooling(h_conv2, 2)  # output size 8x8x64
            h_pool2 = tf.nn.dropout(h_pool2, self.dropout_keep_prob)

            ## conv3 layer ##
            W_conv3 = self.weight_variable([5, 5, 64, 128])  # patch 5x5, in size 64, out size 128
            b_conv3 = self.bias_variable([128])
            h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)  # output size 8x8x128
            h_pool3 = self.max_pooling(h_conv3, 2)  # output size 4x4x128
            h_pool3 = tf.nn.dropout(h_pool3, self.dropout_keep_prob)

            # 采用多层级结构
            # 1st stage output
            h_pool1 = self.max_pooling(h_pool1, 4) # 这里pooling size的为4的原因是,将pool1和pool2和pool3的尺寸对应起来 4*4
            shape = h_pool1.get_shape().as_list()
            h_pool1 = tf.reshape(h_pool1, [-1, shape[1] * shape[2] * shape[3]])

            # 2nd stage output
            h_pool2 = self.max_pooling(h_pool2, 2)
            # pool2 = pool(pool2, size=2)
            shape = h_pool2.get_shape().as_list()
            h_pool2 = tf.reshape(h_pool2, [-1, shape[1] * shape[2] * shape[3]])

            # 3rd stage output
            shape = h_pool3.get_shape().as_list()
            h_pool3 = tf.reshape(h_pool3, [-1, shape[1] * shape[2] * shape[3]])

            # 拼接
            # flattened = tf.concat([h_pool1, h_pool2, h_pool3], 1) # shape=(?,3584), 4*4*32 + 4*4*64 + 4*4*128 = 3584
            flattened = tf.concat([h_pool1, h_pool2, h_pool3], 1)
            shape = flattened.get_shape().as_list()
            ## func1 layer ##
            # W_fc1 = self.weight_variable([4 * 4 * 128, 1024])
            W_fc1 = self.weight_variable([shape[1], 1024])
            b_fc1 = self.bias_variable([1024])
            # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
            # h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
            h_fc1 = tf.nn.relu(tf.matmul(flattened, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.dropout_keep_prob)

            ## func2 layer ##
            W_fc2 = self.weight_variable([1024, num_class])
            b_fc2 = self.bias_variable([num_class])
            self.logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            # self.prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        with tf.name_scope("loss"):
            # 全连接层参数进行 L2 正则化
            regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits + 1e-10, labels=self.input_y)
            self.cost = tf.reduce_mean(self.loss) + l2_reg_lambda * regularizers

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits, 1, name='prediction')
            self.target = tf.argmax(self.input_y, 1, name='target')
            correct_prediction = tf.equal(self.prediction, self.target)
            self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


