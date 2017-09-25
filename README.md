# tensorflow-traffic-signs
Traffic Signs Recognition with Tensorflow and a convolutional network

Introduction:

1.This is my attempt to tackle traffic signs classification problem with a convolutional neural network implemented in TensorFlow.

2.The highlights of this solution would be data preprocessing, data augmentation, and skipping connections in the network.

3.The accuracy on the German Traffic Sign Dataset is reaching 98.82% accuracy at present, and this can still be improved

Data processing and augmentation:

1.I resized all images to 48*48*3, and scaled of pixel values to [0, 1] (as currently they are in [0, 255] range). Then, I will only use a single channel in my model, e.g. grayscale images instead of color ones.

2.Data augmentaion consists of flipping image, resizing and rotationing image, and so on.

3.More details can be found at https://navoshta.com/traffic-signs-classification/

CNN Network:

	

