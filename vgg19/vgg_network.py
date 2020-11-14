import tensorflow as tf
import numpy as np
import scipy.io


class VGG:
    layers_name = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = scipy.io.loadmat(data_path)
        mean = self.data['normalization'][0][0][0]
        self.mean_pixel = np.mean(mean, axis=(0, 1))
        self.weights = self.data['layers'][0]

    def preprocess(self, image):
        return image - self.mean_pixel


    def unprocess(self, image):
        return image + self.mean_pixel

    def net(self, input_image):
        net = {}
        current_layer = input_image
        for i, name in enumerate(self.layers_name):
            if name[:4] == 'conv':
                kernels, bias = self.weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current_layer = self.conv(current_layer, kernels, bias)
            elif name[:4] == 'relu':
                current_layer = tf.nn.relu(current_layer)
            elif name[:4] == 'pool':
                current_layer = self.max_pool(current_layer)
            net[name] = current_layer

        assert len(net) == len(self.layers_name)
        return net

    def conv(self,input, weights, bias):
        conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                padding='SAME')
        return tf.nn.bias_add(conv, bias)

    def max_pool(self,input):
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
