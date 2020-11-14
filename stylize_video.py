# Copyright (c) 2016-2017 Shafeen Tejani. Released under GPLv3.

import os

import numpy as np
from os.path import exists

from argparse import ArgumentParser
import tensorflow as tf
import transform_network
import cv2 as cv

NETWORK_PATH = 'network'
OUTPUT_PATH = "yolo_out_py.mov"


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str,
                        dest='video_path', help='video path',
                        metavar='CONTENT', required=True)

    parser.add_argument('--network-path', type=str,
                        dest='network_path',
                        help='path to network (default %(default)s)',
                        metavar='NETWORK_PATH', default=NETWORK_PATH)

    parser.add_argument('--output-path', type=str,
                        dest='output_path',
                        help='path for output',
                        metavar='OUTPUT_PATH', default=OUTPUT_PATH)

    return parser


def check_opts(opts):
    if opts.video_path != "camera":
        assert exists(opts.video_path), "video not found!"
    assert exists(opts.network_path), "network not found!"


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)

    network_path = options.network_path
    if not os.path.isdir(network_path):
        parser.error("Network %s does not exist." % network_path)

    if options.video_path=="camera":
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(options.video_path)

    # 保存视频流
    vid_writer = cv.VideoWriter(options.output_path, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    with tf.Session() as sess:
        img_placeholder = tf.placeholder(tf.float32, shape=(1,cap.get(4),cap.get(3),3), name='img_placeholder')

        network = transform_network.net(img_placeholder)
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(network_path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")

        while cv.waitKey(1) < 0:
            # 读取一帧
            hasFrame, frame = cap.read()

            frame = np.asarray([frame])
            prediction = sess.run(network, feed_dict={img_placeholder: frame})

            cv.imshow("test",np.clip(prediction[0], 0, 255).astype(np.uint8))
            vid_writer.write(np.clip(prediction[0], 0, 255).astype(np.uint8))
            key = cv.waitKey(1)
            # 按q停止
            if key == ord("q"):
                break

if __name__ == '__main__':
    main()
