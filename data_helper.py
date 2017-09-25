import os
import random
import skimage.data
# import skimage.transform
from skimage.transform import rotate
from skimage.transform import rescale
from skimage import exposure
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gc
import gzip
import random
from pandas.io.parsers import read_csv

class DataHelper(object):

    def load_data(self, data_dir):
        """Loads a data set and returns two lists:

        images: a list of Numpy arrays, each representing an image.
        labels: a list of numbers that represent the images labels.
        """
        # Get all subdirectories of data_dir. Each represents a label.
        directories = [d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))]
        # Loop through the label directories and collect the data in
        # two lists, labels and images.
        labels = []
        images = []
        for d in directories:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir) if f.endswith(".ppm")]
            # For each label, load it's images and add them to the images list.
            # And add the label number (i.e. directory name) to the labels list.
            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(int(d))
        return np.array(images), np.array(labels)

    def load_test_data(self, test_data_dir, test_label_file):
        test_images = []
        # file_names = [os.path.join(test_data_dir, f)
        #               for f in os.listdir(test_data_dir) if f.endswith(".ppm")]
        #
        # // 没有按顺序来
        # # For each label, load it's images and add them to the images list.
        # # And add the label number (i.e. directory name) to the labels list.
        # for f in file_names:
        #     test_images.append(skimage.data.imread(f))
        labels = read_csv(test_label_file).values[:]
        test_labels = []
        for label in labels:
            l = label[0].split(';')
            image_dir = os.path.join(test_data_dir, l[0])
            test_images.append(skimage.data.imread(image_dir))
            test_labels.append(l[-1])

        # 数据预处理
        images32 = [skimage.transform.resize(image, (48, 48))
                    for image in test_images]
        labels_test = np.array(test_labels)
        images_test = np.array(images32)

        images_a = 0.299 * images_test[:, :, :, 0] + 0.587 * images_test[:, :, :, 1] + 0.114 * images_test[:, :, :, 2]
        for i in range(images_a.shape[0]):
            images_a[i] = exposure.equalize_adapthist(images_a[i])
        i_shapes = images_a.shape
        images_a = np.reshape(images_a, (i_shapes[0], i_shapes[1], i_shapes[2], 1))

        return images_a, labels_test

    def data_augmentation_flip(self, X, y):
        # data augmentation, 包括Flipping
        # 对于traffic sign的训练数据,进行翻转, 来增加数据量.
        # 要注意的是有些图片可以水平翻转有些可以竖直翻转,有些翻转后类别不变,有些翻转类别却发生变化,例如左转右转

        # 当水平翻转时，类别不变
        self_flippable_horizontally = np.array([11, 12, 13, 15, 17, 18, 22, 26, 30, 35])
        # 当垂直翻转时，类别不变
        self_flippable_vertically = np.array([1, 5, 12, 15, 17])
        # 当水平翻转然后垂直翻转后,类别不变
        self_flippable_both = np.array([32, 40])
        # 当水平翻转时，变成了其他类别
        cross_flippable = np.array([
            [19, 20],
            [33, 34],
            [36, 37],
            [38, 39],
            [20, 19],
            [34, 33],
            [37, 36],
            [39, 38],
        ])
        num_classes = 43

        X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
        y_extended = np.empty([0], dtype=y.dtype)

        for c in range(num_classes):
            # 首先复制此类的现有数据
            X_extended = np.append(X_extended, X[y == c], axis=0)
            # If we can flip images of this class horizontally and they would still belong to said class...
            if c in self_flippable_horizontally:
                # ...Copy their flipped versions into extended array.
                X_extended = np.append(X_extended, X[y == c][:, :, ::-1, :], axis=0)
            # If we can flip images of this class horizontally and they would belong to other class...
            if c in cross_flippable[:, 0]:
                # ...Copy flipped images of that other class to the extended array.
                flip_class = cross_flippable[cross_flippable[:, 0] == c][0][1]
                X_extended = np.append(X_extended, X[y == flip_class][:, :, ::-1, :], axis=0)
            # Fill labels for added images set to current class.
            y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

            # If we can flip images of this class vertically and they would still belong to said class...
            if c in self_flippable_vertically:
                # ...Copy their flipped versions into extended array.
                X_extended = np.append(X_extended, X[y == c][:, ::-1, :, :], axis=0)
            # Fill labels for added images set to current class.
            y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

            # If we can flip images of this class horizontally AND vertically and they would still belong to said class...
            if c in self_flippable_both:
                # ...Copy their flipped versions into extended array.
                X_extended = np.append(X_extended, X_extended[y_extended == c][:, ::-1, ::-1, :], axis=0)
            # Fill labels for added images set to current class.
            y_extended = np.append(y_extended, np.full((X_extended.shape[0] - y_extended.shape[0]), c, dtype=int))

        return (X_extended, y_extended)

    # def data_augmentation_rotate_rescale(self, X, y):
    #     # 1 随机旋转[-15,15]
    #     # 2 图片尺寸缩放[0.9, 1.1]
    #     # images, labels = self.load_data(training_paths)
    #     # 针对样本不均衡的问题:这里对样本少的类,数据扩充多一点
    #     small_class = [0,6,14,15,16,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,36,37,39,40,41,42]
    #     X_extended = np.empty([0, X.shape[1], X.shape[2], X.shape[3]], dtype=X.dtype)
    #     y_extended = np.empty([0], dtype=y.dtype)
    #     for i in range(X.shape[0]):
    #         if y[i] in small_class:
    #             i_shapes = X[i].shape
    #             newX = np.reshape(X[i], (i_shapes[0], i_shapes[1], i_shapes[2], 1))
    #             X_extended = np.append(X_extended, newX)
    #             y_extended = np.append(y_extended, y[i])
    #             newX= rotate(X[i], random.uniform(-15, 15), mode='edge')
    #             i_shapes = newX.shape
    #             newX = np.reshape(newX, (i_shapes[0], i_shapes[1], i_shapes[2], 1))
    #             # newX = rescale(newX, random.uniform(0.9, 1.1))
    #             X_extended = np.append(X_extended, newX)
    #             y_extended = np.append(y_extended, y[i])
    #             newX = rotate(X[i], random.uniform(-15, 15), mode='edge')
    #             i_shapes = newX.shape
    #             newX = np.reshape(newX, (i_shapes[0], i_shapes[1], i_shapes[2], 1))
    #             # newX = rescale(newX, random.uniform(0.9, 1.1))
    #             X_extended = np.append(X_extended, newX)
    #             y_extended = np.append(y_extended, y[i])
    #         else:
    #             i_shapes = X[i].shape
    #             newX = np.reshape(X[i], (i_shapes[0], i_shapes[1], i_shapes[2], 1))
    #             X_extended = np.append(X_extended, newX)
    #             y_extended = np.append(y_extended, y[i])
    #             newX = rotate(X[i], random.uniform(-15, 15), mode='edge')
    #             i_shapes = newX.shape
    #             newX = np.reshape(newX, (i_shapes[0], i_shapes[1], i_shapes[2], 1))
    #             # newX = rescale(newX, random.uniform(0.9, 1.1))
    #             X_extended = np.append(X_extended, newX)
    #             y_extended = np.append(y_extended, y[i])
    #     return (X_extended, y_extended)

    def data_processing(self, training_paths, image_length, image_width, num_class):
        # 数据预处理
        # 1. Resize images to image_length*image_width
        # 2. Normalizing values to the range 0.0-1.0
        # 3. label 变成[1,0,0,0]的形式 (已修改,将这一步移到最后处理)
        # 4. Convert to grayscale, e.g. single channel Y, 灰度值=0.2989 * R + 0.5870 * G + 0.1140 * B
        # 5. 对比度增强--localized histogram equalization(局部直方图均衡).(有一些图像可能比较模糊，对比度增强可能有很好的效果。对比度增强的方式很多。)

        images, labels = self.load_data(training_paths)
        images32 = [skimage.transform.resize(image, (image_length, image_width))
                    for image in images]
        # 第四步,彩色图变成灰度图
        images_a = np.array(images32)
        images_a = 0.299 * images_a[:, :, :, 0] + 0.587 * images_a[:, :, :, 1] + 0.114 * images_a[:, :, :, 2]
        # 第五步,直方图均衡
        for i in range(images_a.shape[0]):
            images_a[i] = exposure.equalize_adapthist(images_a[i])

        # new_labels = []
        # for label in labels:
        #     new_label = [0] * num_class
        #     new_label[label] = 1
        #     new_labels.append(new_label)
        # labels_a = np.array(new_labels)

        # Add a single grayscale channel
        # images_a = np.reshape(images_a, (i_shapes[0], i_shapes[1], i_shapes[2], 1))
        # i_shapes = images_a.shape
        images_a = images_a.reshape(images_a.shape + (1,))
        labels = np.array(labels)
        return images_a, labels

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        # print(data)
        print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    # Data Preparatopn
    # ==================================================


    def getDataSets(self, training_paths, percent_dev, batch_size, image_length, image_width, num_class):
        # 1. 图片预处理
        X1, Y1 = self.data_processing(training_paths, image_length, image_width, num_class)

        # 2. 图片增广
        x, y = self.data_augmentation_flip(X1, Y1)
        # x, y = self.data_augmentation_rotate_rescale(X1, Y1)

        # 3. label 变成[1,0,0,0]的形式
        new_labels = []
        for label in y:
            new_label = [0] * num_class
            new_label[label] = 1
            new_labels.append(new_label)
        new_y = np.array(new_labels)

        sum_no_of_batches = 0

        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = new_y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100

        # Split train/test set
        # self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
        # TODO: This is very crude, should use cross-validation
        x_train, x_dev = x_shuffled[:dev_idx], x_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
        train_set = (x_train, y_train)
        dev_set = (x_dev, y_dev)
        gc.collect()
        return train_set, dev_set, sum_no_of_batches

# if __name__ == '__main__':
# # Load training and testing datasets.
# ROOT_PATH = "/data/MyTensorflow/MyProject/traffic-signs-master"
# train_data_dir = os.path.join(ROOT_PATH, "traffic-signs-data/train/Final_Training/Images")
# test_data_dir = os.path.join(ROOT_PATH, "traffic-signs-data/test/Final_test/Images")
#
# images, labels = load_data(train_data_dir)
# print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
#
# def display_images_and_labels(images, labels):
#     # 每个label中选取一张图片展示
#     """Display the first image of each label."""
#     unique_labels = set(labels)
#     plt.figure(figsize=(30, 20))
#     i = 1
#     for label in unique_labels:
#         # Pick the first image for each label.
#         image = images[labels.index(label)]
#         plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
#         plt.axis('off')
#         plt.title("Label {0} ({1})".format(label, labels.count(label)))
#         i += 1
#         _ = plt.imshow(image)
#     plt.show()

# display_images_and_labels(images, labels)

# def display_label_images(images, label):
#     """Display images of a specific label."""
#     # 展示某个标签的24张图片
#     limit = 24  # show a max of 24 images
#     plt.figure(figsize=(15, 5))
#     i = 1
#
#     start = labels.index(label)
#     end = start + labels.count(label)
#     for image in images[start:end][:limit]:
#         plt.subplot(3, 8, i)  # 3 rows, 8 per row
#         plt.axis('off')
#         i += 1
#         plt.imshow(image)
#     plt.show()
#
# display_label_images(images, 2)

# 1. Resize images to 32*32
# 2. Normalizing values to the range 0.0-1.0
# images32 = [skimage.transform.resize(image, (32, 32))
#                 for image in images]
# # display_images_and_labels(images32, labels)
# labels_a = np.array(labels)
# images_a = np.array(images32)
# print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)

