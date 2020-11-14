from vgg19 import vgg_network

import tensorflow as tf
import numpy as np
import utils

from sys import stdout
from functools import reduce
import transform_network


class LossCalculator:

    def __init__(self, vgg, target_image):
        self.vgg = vgg
        self.transform_loss_net = vgg.net(vgg.preprocess(target_image))

    def content_loss(self, content_input_batch, content_layer, content_weight):
        print(content_input_batch.dtype)
        content_loss_net = self.vgg.net(self.vgg.preprocess(content_input_batch))
        return content_weight * (2 * tf.nn.l2_loss(
            content_loss_net[content_layer] - self.transform_loss_net[content_layer]) /
                                 (_tensor_size(content_loss_net[content_layer])))

    def style_loss(self, style_image, style_layers, style_weight):
        style_loss=0
        style_image_placeholder = tf.placeholder('float', shape=style_image.shape)
        style_loss_net = self.vgg.net(self.vgg.preprocess(style_image_placeholder))
        with tf.Session() as sess:
            for layer in style_layers:
                image_feature = style_loss_net[layer].eval(feed_dict={style_image_placeholder:style_image})
                image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
                style_image_gram = np.matmul(image_feature.T, image_feature) / image_feature.size

                image_feature = self.transform_loss_net[layer]
                #map(function,iterable,...)参数列表中每一个元素调用function，返回function返回值新列表
                batch_size, height, width, number = map(lambda i: i.value, image_feature.get_shape())
                size = height * width * number
                image_feature = tf.reshape(image_feature, (batch_size, height * width, number))
                input_image_gram = tf.matmul(tf.transpose(image_feature, perm=[0, 2, 1]), image_feature) / size

                style_loss += (2 * tf.nn.l2_loss(input_image_gram - style_image_gram) / style_image_gram.size)

            return style_weight * (style_loss)

    def tv_loss(self, image, tv_weight):
        # total variation denoising
        shape = tuple(image.get_shape().as_list())
        tv_y_size = _tensor_size(image[:, 1:, :, :])
        tv_x_size = _tensor_size(image[:, :, 1:, :])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
                 tv_y_size) +
                (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
                 tv_x_size))

        return tv_loss


class FastStyleTransfer:
    CONTENT_LAYER = 'relu3_3'
    STYLE_LAYERS = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')

    def __init__(self, vgg_path, style_image, content_weight, style_weight, tv_weight, batch_size):

        vgg = vgg_network.VGG(vgg_path)
        self.style_image = style_image
        self.batch_size = batch_size

        self.input_batch = tf.placeholder(tf.float32,
                                          shape=(batch_size, 256, 256, 3),
                                          name="input_batch")

        self.target_image = transform_network.net(self.input_batch)

        loss_calculator = LossCalculator(vgg, self.target_image)

        self.content_loss = loss_calculator.content_loss(
            self.input_batch,
            self.CONTENT_LAYER,
            content_weight) / self.batch_size

        self.style_loss = loss_calculator.style_loss(
            self.style_image,
            self.STYLE_LAYERS,
            style_weight) / self.batch_size

        self.total_variation_loss = loss_calculator.tv_loss(
            self.target_image,
            tv_weight) / self.batch_size

        self.loss = self.content_loss + self.style_loss + self.total_variation_loss

    def _current_loss(self, feed_dict):
        losses = {}
        losses['content'] = self.content_loss.eval(feed_dict=feed_dict)
        losses['style'] = self.style_loss.eval(feed_dict=feed_dict)
        losses['total_variation'] = self.total_variation_loss.eval(feed_dict=feed_dict)
        losses['total'] = self.loss.eval(feed_dict=feed_dict)
        return losses

    def train(self, content_training_images, learning_rate, epochs, checkpoint_iterations):

        train_step = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method='L-BFGS-B', options={'maxiter': epochs,'disp': 0})

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iterations = 0
            for epoch in range(epochs):
                for i in range(0, len(content_training_images), self.batch_size):

                    stdout.write('Iteration %d\n' % (iterations + 1))

                    batch = self._load_batch(content_training_images[i: i + self.batch_size])

                    train_step.minimize(sess, feed_dict={self.input_batch: batch})

                    if iterations % checkpoint_iterations == 0:
                        # 生成器，返回一个元组，下次从断点处继续执行
                        yield (
                            iterations,
                            sess,
                            self.target_image.eval(feed_dict={self.input_batch: batch})[0],
                            self._current_loss({self.input_batch: batch})
                        )
                    iterations += 1

    def _load_batch(self, image_paths):
        batch_imgs = []
        for batch_file in image_paths:
            img = utils.load_image(batch_file, (256, 256))
            batch_imgs.append(img)

        return np.asarray(batch_imgs)


def _tensor_size(tensor):
    from operator import mul
    '''
    将一个数据集合（链表，元组等）中的所有数据进行下列操作：
    用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，
    得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。
    '''
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)