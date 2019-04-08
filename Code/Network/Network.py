"""
@file    Network.py
@author  rohithjayarajan
@date 02/22/2019

Licensed under the
GNU General Public License v3.0
"""

import tensorflow as tf
import sys
import numpy as np
# Don't generate pyc codes
sys.dont_write_bytecode = True


class SpeedNetwork:

    def SpeedNetVGG(self, Img, isTrain):
        """
        Inputs: 
        Img is a MiniBatch of the current image
        ImageSize - Size of the Image
        Outputs:
        prLogits - logits output of the network
        prSoftMax - softmax output of the network
        """
        # Conv1
        with tf.variable_scope("ConvolutionalBlock1"):
            conv1 = tf.layers.conv2d(
                inputs=Img,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv1')
            bn1 = tf.layers.batch_normalization(conv1)
            z1 = tf.nn.relu(bn1, name='ReLU1')

        with tf.variable_scope("ConvolutionalBlock2"):
            conv2 = tf.layers.conv2d(
                inputs=z1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv2')
            bn2 = tf.layers.batch_normalization(conv2)
            z2 = tf.nn.relu(bn2, name='ReLU2')

        # MaxPool
        pool1 = tf.layers.max_pooling2d(
            z2,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool1')

        # Conv2
        with tf.variable_scope("ConvolutionalBlock3"):
            conv3 = tf.layers.conv2d(
                inputs=pool1,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                name='conv3')
            bn3 = tf.layers.batch_normalization(conv3)
            z3 = tf.nn.relu(bn3, name='ReLU3')

        with tf.variable_scope("ConvolutionalBlock4"):
            conv4 = tf.layers.conv2d(
                inputs=z3,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                name='conv4')
            bn4 = tf.layers.batch_normalization(conv4)
            z4 = tf.nn.relu(bn4, name='ReLU4')

        # MaxPool
        pool2 = tf.layers.max_pooling2d(
            z4,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool2')

        # Conv3
        with tf.variable_scope("ConvolutionalBlock5"):
            conv5 = tf.layers.conv2d(
                inputs=pool2,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                name='conv5')
            bn5 = tf.layers.batch_normalization(conv5)
            z5 = tf.nn.relu(bn5, name='ReLU5')

        with tf.variable_scope("ConvolutionalBlock6"):
            conv6 = tf.layers.conv2d(
                inputs=z5,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                name='conv6')
            bn6 = tf.layers.batch_normalization(conv6)
            z6 = tf.nn.relu(bn6, name='ReLU6')

        with tf.variable_scope("ConvolutionalBlock7"):
            conv7 = tf.layers.conv2d(
                inputs=z6,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                name='conv7')
            bn7 = tf.layers.batch_normalization(conv7)
            z7 = tf.nn.relu(bn7, name='ReLU7')

        # MaxPool
        pool3 = tf.layers.max_pooling2d(
            z7,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool3')

        # Conv4
        with tf.variable_scope("ConvolutionalBlock8"):
            conv8 = tf.layers.conv2d(
                inputs=pool3,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                name='conv8')
            bn8 = tf.layers.batch_normalization(conv8)
            z8 = tf.nn.relu(bn8, name='ReLU8')

        with tf.variable_scope("ConvolutionalBlock9"):
            conv9 = tf.layers.conv2d(
                inputs=z8,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                name='conv9')
            bn9 = tf.layers.batch_normalization(conv9)
            z9 = tf.nn.relu(bn9, name='ReLU9')

        with tf.variable_scope("ConvolutionalBlock10"):
            conv10 = tf.layers.conv2d(
                inputs=z9,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                name='conv10')
            bn10 = tf.layers.batch_normalization(conv10)
            z10 = tf.nn.relu(bn10, name='ReLU10')

        # MaxPool
        pool4 = tf.layers.max_pooling2d(
            z10,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool4')

        # Conv5
        with tf.variable_scope("ConvolutionalBlock11"):
            conv11 = tf.layers.conv2d(
                inputs=pool4,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                name='conv11')
            bn11 = tf.layers.batch_normalization(conv11)
            z11 = tf.nn.relu(bn11, name='ReLU11')

        with tf.variable_scope("ConvolutionalBlock12"):
            conv12 = tf.layers.conv2d(
                inputs=z11,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                name='conv12')
            bn12 = tf.layers.batch_normalization(conv12)
            z12 = tf.nn.relu(bn12, name='ReLU12')

        with tf.variable_scope("ConvolutionalBlock13"):
            conv13 = tf.layers.conv2d(
                inputs=z12,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                name='conv13')
            bn13 = tf.layers.batch_normalization(conv13)
            z13 = tf.nn.relu(bn13, name='ReLU13')

        # MaxPool
        pool5 = tf.layers.max_pooling2d(
            z13,
            pool_size=[2, 2],
            strides=2,
            padding='valid',
            name='pool5')

        # FC-1024
        z14_flat = tf.layers.flatten(pool5)
        dense1 = tf.layers.dense(inputs=z14_flat, units=1024, activation=None)
        bn14 = tf.layers.batch_normalization(dense1)
        z14 = tf.nn.relu(bn14, name='ReLU14')

        dropout2 = tf.layers.dropout(inputs=z14, rate=0.5, training=isTrain)

        # FC-1024
        dense2 = tf.layers.dense(inputs=dropout2, units=1024, activation=None)
        bn15 = tf.layers.batch_normalization(dense2)
        z15 = tf.nn.relu(bn15, name='ReLU15')

        dropout3 = tf.layers.dropout(inputs=z15, rate=0.5, training=isTrain)

        Speed = tf.layers.dense(inputs=dropout3, units=1, activation=None)

        return Speed

    def SpeedNetNVIDIAe2e(self, Img, isTrain):
        """
        Inputs: 
        Img is a MiniBatch of the current image
        ImageSize - Size of the Image
        Outputs:
        prLogits - logits output of the network
        prSoftMax - softmax output of the network
        """
        # Img = tf.image.per_image_standardization(Img)

        # Conv1
        with tf.variable_scope("ConvolutionalBlock1"):
            conv1 = tf.layers.conv2d(
                inputs=Img,
                filters=24,
                kernel_size=[5, 5],
                strides=(2, 2),
                padding="same",
                name='conv1')
            bn1 = tf.layers.batch_normalization(conv1)
            z1 = tf.nn.relu(bn1, name='ReLU1')

        # Conv2
        with tf.variable_scope("ConvolutionalBlock2"):
            conv2 = tf.layers.conv2d(
                inputs=z1,
                filters=36,
                kernel_size=[5, 5],
                strides=(2, 2),
                padding="same",
                name='conv2')
            bn2 = tf.layers.batch_normalization(conv2)
            z2 = tf.nn.relu(bn2, name='ReLU2')

        # Conv3
        with tf.variable_scope("ConvolutionalBlock3"):
            conv3 = tf.layers.conv2d(
                inputs=z2,
                filters=48,
                kernel_size=[5, 5],
                strides=(2, 2),
                padding="same",
                name='conv3')
            bn3 = tf.layers.batch_normalization(conv3)
            z3 = tf.nn.relu(bn3, name='ReLU3')

        # Conv4
        with tf.variable_scope("ConvolutionalBlock4"):
            conv4 = tf.layers.conv2d(
                inputs=z3,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv4')
            bn4 = tf.layers.batch_normalization(conv4)
            z4 = tf.nn.relu(bn4, name='ReLU4')

        # Conv5
        with tf.variable_scope("ConvolutionalBlock5"):
            conv5 = tf.layers.conv2d(
                inputs=z4,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                name='conv5')
            bn5 = tf.layers.batch_normalization(conv5)
            z5 = tf.nn.relu(bn5, name='ReLU5')

        # FC-1164
        z6_flat = tf.layers.flatten(z5)
        dense1 = tf.layers.dense(inputs=z6_flat, units=1164, activation=None)
        bn6 = tf.layers.batch_normalization(dense1)
        z6 = tf.nn.relu(bn6, name='ReLU6')

        dropout1 = tf.layers.dropout(inputs=z6, rate=0.5, training=isTrain)

        # FC-100
        dense2 = tf.layers.dense(inputs=dropout1, units=100, activation=None)
        bn7 = tf.layers.batch_normalization(dense2)
        z7 = tf.nn.relu(bn7, name='ReLU7')

        dropout2 = tf.layers.dropout(inputs=z7, rate=0.5, training=isTrain)

        # FC-50
        dense3 = tf.layers.dense(inputs=dropout2, units=50, activation=None)
        bn8 = tf.layers.batch_normalization(dense3)
        z8 = tf.nn.relu(bn8, name='ReLU8')

        # FC-10
        dense4 = tf.layers.dense(inputs=z8, units=10, activation=None)
        bn9 = tf.layers.batch_normalization(dense4)
        z9 = tf.nn.relu(bn9, name='ReLU9')

        Speed = tf.layers.dense(inputs=z9, units=1, activation=None)
        return Speed
