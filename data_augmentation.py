from skimage.transform import rotate
from skimage.transform import rescale
from skimage import exposure, io
import skimage.data
import numpy as np
import random
import os

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

def data_augmen(data_dir):
    # 1 随机旋转[-15,15]
    # 2 图片尺寸缩放[0.9, 1.1]
    # images, labels = self.load_data(training_paths)
    # 针对样本不均衡的问题:这里对样本少的类,数据扩充多一点
    small_class = [0, 6, 14, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40,
                   41, 42]
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        im_index = 0
        for f in file_names:
            image = skimage.data.imread(f)

            if int(d) in small_class:
                newX = rotate(image, random.uniform(-20, 20), mode='edge')
                newX = rescale(newX, random.uniform(0.9, 1.1))
                im_name = 'aug_' + str(im_index) + '.ppm'
                im_index += 1
                savename = os.path.join(label_dir, im_name)
                io.imsave(savename, newX)

                newX = rotate(image, random.uniform(-20, 20), mode='edge')
                newX = rescale(newX, random.uniform(0.9, 1.1))
                im_name = 'aug_' + str(im_index) + '.ppm'
                im_index += 1
                savename = os.path.join(label_dir, im_name)
                io.imsave(savename, newX)

                newX = rotate(image, random.uniform(-20, 20), mode='edge')
                newX = rescale(newX, random.uniform(0.9, 1.1))
                im_name = 'aug_' + str(im_index) + '.ppm'
                im_index += 1
                savename = os.path.join(label_dir, im_name)
                io.imsave(savename, newX)
            else:
                newX = rotate(image, random.uniform(-20, 20), mode='edge')
                newX = rescale(newX, random.uniform(0.9, 1.1))
                im_name = 'aug_' + str(im_index) + '.ppm'
                im_index += 1
                savename = os.path.join(label_dir, im_name)
                io.imsave(savename, newX)
        # print(im_index)

data_augmen('../traffic-signs-master/traffic-signs-data/train/Final_Training/Images')
