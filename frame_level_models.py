# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags
import tensorflow.contrib as tf_contrib
weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None
weight_regularizer_fully = None

import scipy.io as sio
import numpy as np

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

FLAGS = flags.FLAGS

flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")

flags.DEFINE_integer("iterations", 40,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", False,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')

flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                     "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                     "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                  "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")

flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")

flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                  "ReLU after the NetFV hidden layer.")

flags.DEFINE_bool("fv_couple_weights", True,
                  "Coupling cluster weights or not")

flags.DEFINE_float("fv_coupling_factor", 0.01,
                   "Coupling factor")

flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")

flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")
flags.DEFINE_integer("gru_cells_audio", 128, "Number of GRU cells (audio).")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                  "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                  "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")

flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")


class LightVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        vlad = tf.matmul(activation, reshaped_input)

        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad


class NetVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        #print("====================NetVLAD===================")
        #print("input:{}".format(reshaped_input.shape.as_list()))

        #input:[40,1024]
        #cluster_w:[1024,256]
        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))
        #print("cluster_w:{}".format(cluster_weights.shape.as_list()))

        #tensorboard可视化
        tf.summary.histogram("cluster_weights", cluster_weights)
        #40x256
        activation = tf.matmul(reshaped_input, cluster_weights)
        #print("active1:{}".format(activation.shape.as_list()))
        #print("add_batch_norm:{}".format(self.add_batch_norm))

        #true
        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases
        #[40,256]
        activation = tf.nn.softmax(activation)
        #print("softmax:{}".format(activation.shape.as_list()))
        tf.summary.histogram("cluster_output", activation)

        #[1,40,256]
        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
        #print("reshape:{}".format(activation.shape.as_list()))
        #[1,1,256]
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)
        #print("a_sum:{}".format(a_sum.shape.as_list()))

        #[1,1024,256]
        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, self.feature_size, self.cluster_size],
                                           initializer=tf.random_normal_initializer(
                                               stddev=1 / math.sqrt(self.feature_size)))
        #print("cluster_w2:{}".format(cluster_weights2.shape.as_list()))
        #[1,1024,256] multiply元素级别相乘
        a = tf.multiply(a_sum, cluster_weights2)
        #print("a:{}".format(a.shape.as_list()))

        #[1,256,40]
        activation = tf.transpose(activation, perm=[0, 2, 1])
        #print("transpose:{}".format(activation.shape.as_list()))

        #[1,40,1024]
        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        #1,256,1024
        vlad = tf.matmul(activation, reshaped_input)
        #print("vlad1::{}".format(vlad.shape.as_list()))
        #1,1024,256
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        #print("vld_trans:{}".format(vlad.shape.as_list()))
        #1.11024,256
        vlad = tf.subtract(vlad, a)
        #print("vlad_subtract:{}".format(vlad.shape.as_list()))
        #1,1024,256
        vlad = tf.nn.l2_normalize(vlad, 1)
        #print("vlad_l2_norm:{}".format(vlad.shape.as_list()))
        #1,1024*256
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
        #print("vlad_reshape:{}".format(vlad.shape.as_list()))
        vlad = tf.nn.l2_normalize(vlad, 1)
        # print("output:{}".format(vlad.shape.as_list()))
        # print("=====================================")
        #output:[1, 262144]

        return vlad


class NetVLAGD():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        gate_weights = tf.get_variable("gate_weights",
                                       [1, self.cluster_size, self.feature_size],
                                       initializer=tf.random_normal_initializer(
                                           stddev=1 / math.sqrt(self.feature_size)))

        gate_weights = tf.sigmoid(gate_weights)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])

        vlagd = tf.matmul(activation, reshaped_input)
        vlagd = tf.multiply(vlagd, gate_weights)

        vlagd = tf.transpose(vlagd, perm=[0, 2, 1])

        vlagd = tf.nn.l2_normalize(vlagd, 1)

        vlagd = tf.reshape(vlagd, [-1, self.cluster_size * self.feature_size])
        vlagd = tf.nn.l2_normalize(vlagd, 1)

        return vlagd


class GatedDBoF():
    def __init__(self, feature_size, max_frames, cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
                                          [feature_size, cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation, 1)

        activation_max = tf.reduce_max(activation, 1)
        activation_max = tf.nn.l2_normalize(activation_max, 1)

        dim_red = tf.get_variable("dim_red",
                                  [cluster_size, feature_size],
                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        cluster_weights_2 = tf.get_variable("cluster_weights_2",
                                            [feature_size, cluster_size],
                                            initializer=tf.random_normal_initializer(
                                                stddev=1 / math.sqrt(feature_size)))

        tf.summary.histogram("cluster_weights_2", cluster_weights_2)

        activation = tf.matmul(activation_max, dim_red)
        activation = tf.matmul(activation, cluster_weights_2)

        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="cluster_bn_2")
        else:
            cluster_biases = tf.get_variable("cluster_biases_2",
                                             [cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
            tf.summary.histogram("cluster_biases_2", cluster_biases)
            activation += cluster_biases

        activation = tf.sigmoid(activation)

        activation = tf.multiply(activation, activation_sum)
        activation = tf.nn.l2_normalize(activation, 1)

        return activation


class SoftDBoF():
    def __init__(self, feature_size, max_frames, cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

        cluster_weights = tf.get_variable("cluster_weights",
                                          [feature_size, cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation, 1)
        activation_sum = tf.nn.l2_normalize(activation_sum, 1)

        if max_pool:
            activation_max = tf.reduce_max(activation, 1)
            activation_max = tf.nn.l2_normalize(activation_max, 1)
            activation = tf.concat([activation_sum, activation_max], 1)
        else:
            activation = activation_sum

        return activation


class DBoF():
    def __init__(self, feature_size, max_frames, cluster_size, activation, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.activation = activation

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training

        cluster_weights = tf.get_variable("cluster_weights",
                                          [feature_size, cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)

        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        if activation == 'glu':
            space_ind = range(cluster_size / 2)
            gate_ind = range(cluster_size / 2, cluster_size)

            gates = tf.sigmoid(activation[:, gate_ind])
            activation = tf.multiply(activation[:, space_ind], gates)

        elif activation == 'relu':
            activation = tf.nn.relu6(activation)

        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        avg_activation = utils.FramePooling(activation, 'average')
        avg_activation = tf.nn.l2_normalize(avg_activation, 1)

        max_activation = utils.FramePooling(activation, 'max')
        max_activation = tf.nn.l2_normalize(max_activation, 1)

        return tf.concat([avg_activation, max_activation], 1)


class NetFV():
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.feature_size, self.cluster_size],
                                          initializer=tf.random_normal_initializer(
                                              stddev=1 / math.sqrt(self.feature_size)))

        covar_weights = tf.get_variable("covar_weights",
                                        [self.feature_size, self.cluster_size],
                                        initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
                                            self.feature_size)))

        covar_weights = tf.square(covar_weights)
        eps = tf.constant([1e-6])
        covar_weights = tf.add(covar_weights, eps)

        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if self.add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [self.cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases

        activation = tf.nn.softmax(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

        if not FLAGS.fv_couple_weights:
            cluster_weights2 = tf.get_variable("cluster_weights2",
                                               [1, self.feature_size, self.cluster_size],
                                               initializer=tf.random_normal_initializer(
                                                   stddev=1 / math.sqrt(self.feature_size)))
        else:
            cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor, cluster_weights)

        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
        fv1 = tf.matmul(activation, reshaped_input)

        fv1 = tf.transpose(fv1, perm=[0, 2, 1])

        # computing second order FV
        a2 = tf.multiply(a_sum, tf.square(cluster_weights2))

        b2 = tf.multiply(fv1, cluster_weights2)
        fv2 = tf.matmul(activation, tf.square(reshaped_input))

        fv2 = tf.transpose(fv2, perm=[0, 2, 1])
        fv2 = tf.add_n([a2, fv2, tf.scalar_mul(-2, b2)])

        fv2 = tf.divide(fv2, tf.square(covar_weights))
        fv2 = tf.subtract(fv2, a_sum)

        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])

        fv2 = tf.nn.l2_normalize(fv2, 1)
        fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])
        fv2 = tf.nn.l2_normalize(fv2, 1)

        fv1 = tf.subtract(fv1, a)
        fv1 = tf.divide(fv1, covar_weights)

        fv1 = tf.nn.l2_normalize(fv1, 1)
        fv1 = tf.reshape(fv1, [-1, self.cluster_size * self.feature_size])
        fv1 = tf.nn.l2_normalize(fv1, 1)

        return tf.concat([fv1, fv2], 1)

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x

def hw_flatten(x) :
    print("hw_flattn:{}".format(type(x)))
    print("hw_flatten:{}".format(x.shape.as_list()))
    return tf.reshape(x, [x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]])
class Attn():

    def __init__(self, channels, scope='attention', sn=False):
        self.channels = channels
        self.scope = scope
        self.sn = sn


    def forward( self, x):
        #tf_variable_scope 共享变量/模型封装
        with tf.variable_scope(self.scope):
            print("x-shape:{}".format(x.shape.as_list()))
            #x = tf.reshape(x, [1, 1, 40, x.shape[1]])
            # f = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            # g = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']
            # h = conv(x, self.channels, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            f = tf.transpose(x, perm=[1, 0])
            g = x
            h = x

            # N = h * w
            s = tf.matmul(g, f)  # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, h)  # [bs, N, C]
            print("!!oo:{}".format(o.shape.as_list()))
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            #o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            print("!!o:{}".format(o.shape.as_list()))
            o_w = tf.get_variable("o_w",[1024, 1024],
                                            initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
                                                1024)))
            #o = conv(o, self.channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            o = tf.matmul(o,o_w)

            x = gamma * o + x
            print("out:{}".format(x.shape.as_list()))

        return x

class NetVLADModelLF(models.BaseModel):
    """Creates a NetVLAD based model.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     imp=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.netvlad_cluster_size
        hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
        relu = FLAGS.netvlad_relu
        dimred = FLAGS.netvlad_dimred
        gating = FLAGS.gating
        remove_diag = FLAGS.gating_remove_diag
        lightvlad = FLAGS.lightvlad
        vlagd = FLAGS.vlagd
        print("=============")

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        print("model_input(before):", model_input)
        print("ranndom_frames:{}".format(random_frames))
        #random_frames=false
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
            print("model_input(After):", model_input)

        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        print("model_input:{}".format(model_input.shape.as_list()))
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        print("model_input reshape:{}".format(reshaped_input.shape.as_list()))

        if lightvlad:
            video_NetVLAD = LightVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
            # audio_NetVLAD = LightVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
        elif vlagd:
            video_NetVLAD = NetVLAGD(1024, max_frames, cluster_size, add_batch_norm, is_training)
            # audio_NetVLAD = NetVLAGD(128,max_frames,cluster_size/2, add_batch_norm, is_training)
        else:
            video_NetVLAD = NetVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
            # audio_NetVLAD = NetVLAD(128,max_frames,cluster_size/2, add_batch_norm, is_training)

        if add_batch_norm:  # and not lightvlad:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")
        ##-------------------------------improve part--------------------------------
        print("-----------------NetVLAD improve---------------------")
        print("reshaped_input:{}".format(reshaped_input.shape.as_list()))
        src_input = reshaped_input[:, 0:1024]
        print("src_input:{}".format(src_input.shape.as_list()))
        #temporal relation
        #[40,1024]
        attention = Attn(1024)
        tem_rla = attention.forward(src_input)

        print("-----------------------------------------------")
        #---------------------------------------------------------------------------


        with tf.variable_scope("video_VLAD"):
            #reshaped_input[:, 0:1024]
            vlad_video = video_NetVLAD.forward(tem_rla)

        # with tf.variable_scope("audio_VLAD"):
        #     vlad_audio = audio_NetVLAD.forward(reshaped_input[:,1024:])

        # vlad = tf.concat([vlad_video, vlad_audio],1)
        vlad = tf.concat([vlad_video], 1)

        vlad_dim = vlad.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

        activation = tf.matmul(vlad, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")

        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [hidden1_size, hidden1_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(hidden1_size)))

            gates = tf.matmul(activation, gating_weights)

            if remove_diag:
                # removes diagonals coefficients
                diagonals = tf.matrix_diag_part(gating_weights)
                gates = gates - tf.multiply(diagonals, activation)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [cluster_size],
                                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)

            activation = tf.multiply(activation, gates)

        #multi-label prediction
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class DbofModelLF(models.BaseModel):
    """Creates a Deep Bag of Frames model.

    The model projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then
    uses a configurable video-level model to classify the now aggregated features.

    The model will randomly sample either frames or sequences of frames during
    training to speed up convergence.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.dbof_cluster_size
        hidden1_size = hidden_size or FLAGS.dbof_hidden_size
        relu = FLAGS.dbof_relu
        cluster_activation = FLAGS.dbof_activation

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)

        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.summary.histogram("input_hist", reshaped_input)

        if cluster_activation == 'glu':
            cluster_size = 2 * cluster_size

        video_Dbof = DBoF(1024, max_frames, cluster_size, cluster_activation, add_batch_norm, is_training)
        audio_Dbof = DBoF(128, max_frames, cluster_size / 8, cluster_activation, add_batch_norm, is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_DBOF"):
            dbof_video = video_Dbof.forward(reshaped_input[:, 0:1024])

        with tf.variable_scope("audio_DBOF"):
            dbof_audio = audio_Dbof.forward(reshaped_input[:, 1024:])

        dbof = tf.concat([dbof_video, dbof_audio], 1)

        dbof_dim = dbof.get_shape().as_list()[1]

        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [dbof_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(dbof, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")
        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            **unused_params)


class GatedDbofModelLF(models.BaseModel):
    """Creates a Gated Deep Bag of Frames model.

    The model projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then
    uses a configurable video-level model to classify the now aggregated features.

    The model will randomly sample either frames or sequences of frames during
    training to speed up convergence.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.dbof_cluster_size
        hidden1_size = hidden_size or FLAGS.dbof_hidden_size
        fc_dimred = FLAGS.fc_dimred
        relu = FLAGS.dbof_relu
        max_pool = FLAGS.softdbof_maxpool

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        print("model_input(Before):", model_input)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
            print("model_input(After):", model_input)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.summary.histogram("input_hist", reshaped_input)

        video_Dbof = GatedDBoF(1024, max_frames, cluster_size, max_pool, add_batch_norm, is_training)
        # audio_Dbof = SoftDBoF(128, max_frames, cluster_size / 8, max_pool, add_batch_norm, is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_DBOF"):
            dbof_video = video_Dbof.forward(reshaped_input[:, 0:1024])

        # with tf.variable_scope("audio_DBOF"):
        #     dbof_audio = audio_Dbof.forward(reshaped_input[:, 1024:])

        # dbof = tf.concat([dbof_video, dbof_audio], 1)
        dbof = tf.concat([dbof_video], 1)
        dbof_dim = dbof.get_shape().as_list()[1]

        if fc_dimred:
            hidden1_weights = tf.get_variable("hidden1_weights",
                                              [dbof_dim, hidden1_size],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(cluster_size)))
            tf.summary.histogram("hidden1_weights", hidden1_weights)
            activation = tf.matmul(dbof, hidden1_weights)

            if add_batch_norm and relu:
                activation = slim.batch_norm(
                    activation,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="hidden1_bn")
            else:
                hidden1_biases = tf.get_variable("hidden1_biases",
                                                 [hidden1_size],
                                                 initializer=tf.random_normal_initializer(stddev=0.01))
                tf.summary.histogram("hidden1_biases", hidden1_biases)
                activation += hidden1_biases

            if relu:
                activation = tf.nn.relu6(activation)
            tf.summary.histogram("hidden1_output", activation)
        else:
            activation = dbof

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class SoftDbofModelLF(models.BaseModel):
    """Creates a Soft Deep Bag of Frames model.

    The model projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then
    uses a configurable video-level model to classify the now aggregated features.

    The model will randomly sample either frames or sequences of frames during
    training to speed up convergence.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.dbof_cluster_size
        hidden1_size = hidden_size or FLAGS.dbof_hidden_size
        fc_dimred = FLAGS.fc_dimred
        relu = FLAGS.dbof_relu
        max_pool = FLAGS.softdbof_maxpool

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        print("model_input(Before):", model_input)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
            print("model_input(After):", model_input)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.summary.histogram("input_hist", reshaped_input)

        video_Dbof = SoftDBoF(1024, max_frames, cluster_size, max_pool, add_batch_norm, is_training)
        # audio_Dbof = SoftDBoF(128, max_frames, cluster_size / 8, max_pool, add_batch_norm, is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_DBOF"):
            dbof_video = video_Dbof.forward(reshaped_input[:, 0:1024])

        # with tf.variable_scope("audio_DBOF"):
        #     dbof_audio = audio_Dbof.forward(reshaped_input[:, 1024:])
        #
        # dbof = tf.concat([dbof_video, dbof_audio], 1)
        dbof = tf.concat([dbof_video], 1)

        dbof_dim = dbof.get_shape().as_list()[1]

        if fc_dimred:
            hidden1_weights = tf.get_variable("hidden1_weights",
                                              [dbof_dim, hidden1_size],
                                              initializer=tf.random_normal_initializer(
                                                  stddev=1 / math.sqrt(cluster_size)))
            tf.summary.histogram("hidden1_weights", hidden1_weights)
            activation = tf.matmul(dbof, hidden1_weights)

            if add_batch_norm and relu:
                activation = slim.batch_norm(
                    activation,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="hidden1_bn")
            else:
                hidden1_biases = tf.get_variable("hidden1_biases",
                                                 [hidden1_size],
                                                 initializer=tf.random_normal_initializer(stddev=0.01))
                tf.summary.histogram("hidden1_biases", hidden1_biases)
                activation += hidden1_biases

            if relu:
                activation = tf.nn.relu6(activation)
            tf.summary.histogram("hidden1_output", activation)
        else:
            activation = dbof

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class AttnLSTM():

    def __init__(self, channels, scope='attention', sn=False):
        self.channels = channels
        self.scope = scope
        self.sn = sn


    def forward( self, x):
        #tf_variable_scope 共享变量/模型封装
        with tf.variable_scope(self.scope):
            print("x-shape:{}".format(x.shape.as_list()))
            #x:[None,40,1024]
            #x = tf.reshape(x, [1, 1, 40, x.shape[1]])
            # f = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            # g = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']
            # h = conv(x, self.channels, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            f = tf.transpose(x, perm=[0,2, 1])#None,1024,40
            g = x
            h = x

            # N = h * w
            s = tf.matmul(g, f)  # # [None, 40, 40]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, h)  # [None, 40, 1024]
            print("!!oo:{}".format(o.shape.as_list()))
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            #o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            print("!!o:{}".format(o.shape.as_list()))
            o_w = tf.get_variable("o_w",[1024, 1024],
                                            initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
                                                1024)))
            #o = conv(o, self.channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            o = tf.matmul(o,o_w)

            x = gamma * o + x
            print("out:{}".format(x.shape.as_list()))

        return x

class AttnLSTM_localTem():

    def __init__(self, channels, scope='attention', sn=False):
        self.channels = channels
        self.scope = scope
        self.sn = sn


    def forward( self, x):
        #tf_variable_scope 共享变量/模型封装
        with tf.variable_scope(self.scope):
            print("x-shape:{}".format(x.shape.as_list()))
            #x:[None,40,1024]
            #x = tf.reshape(x, [1, 1, 40, x.shape[1]])
            # f = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            # g = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']
            # h = conv(x, self.channels, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            x0 = x
            x = tf.expand_dims(x,axis=2)#none,40,1,1024
            x = tf.reshape(x,[-1,40,16,64])
            print("!!spatial-x:{}".format(x.shape.as_list()))
            x = tf.reshape(x,[-1,40,1024])
            print("!!spatial-x1:{}".format(x.shape.as_list()))
            # f = tf.transpose(x, perm=[0,1, 3,2])#None,40,1024,1
            # g = x
            # h = x
            #
            # # N = h * w
            # s = tf.matmul(g, f)  # # [None, 40, 40]
            #
            # beta = tf.nn.softmax(s)  # attention map
            #
            # o = tf.matmul(beta, h)  # [None, 40, 1024]
            # print("!!oo:{}".format(o.shape.as_list()))
            # gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
            #
            # #o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
            # print("!!o:{}".format(o.shape.as_list()))
            # o_w = tf.get_variable("o_w",[1024, 1024],
            #                                 initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
            #                                     1024)))
            # #o = conv(o, self.channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            # o = tf.matmul(o,o_w)
            #
            # x = gamma * o + x
            # print("out:{}".format(x.shape.as_list()))

        return x

class AttnLSTM_spa():

    def __init__(self, channels, scope='attn_spa', sn=False):
        self.channels = channels
        self.scope = scope
        self.sn = sn


    def forward( self, x):
        #tf_variable_scope 共享变量/模型封装
        with tf.variable_scope(self.scope):
            print("x-shape:{}".format(x.shape.as_list()))
            #x:[None,40,1024]
            #x = tf.reshape(x, [1, 1, 40, x.shape[1]])
            # f = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv')  # [bs, h, w, c']
            # g = conv(x, self.channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv')  # [bs, h, w, c']
            # h = conv(x, self.channels, kernel=1, stride=1, sn=self.sn, scope='h_conv')  # [bs, h, w, c]
            x0 = x
            ave_mean = tf.reduce_mean(x,axis=2,keep_dims=True)#None,40,1

            o = tf.multiply(x, ave_mean)  # # [None, 40, 40]

            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o_w = tf.get_variable("o_w",[1024, 1024],
                                            initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
                                                1024)))
            # #o = conv(o, self.channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')
            o = tf.matmul(o,o_w)
            print("03:{}".format(o.shape.as_list()))
            #
            x = gamma * o + x0
            print("x-out:{}".format(x.shape.as_list()))

        return x


class LstmModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        print("========================LSTM==================")
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        random_frames = FLAGS.lstm_random_sequence
        iterations = FLAGS.iterations
        backward = FLAGS.lstm_backward
        print("lstm_size:{}".format(lstm_size))
        print("number_of_layers:{}".format(number_of_layers))

        if random_frames:
            num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
            model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                                   iterations)
        if backward:
            model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
            ], state_is_tuple=False)

        # 前向运动特征编码LSTM
        forward_motion_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
            ], state_is_tuple=False)

        # 反向运动特征编码LSTM
        back_motion_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False)
                for _ in range(number_of_layers)
            ], state_is_tuple=False)

        loss = 0.0
        #----------------------improvement----------------------
        x = model_input
        attention = AttnLSTM(1024)
        out = attention.forward(x)
        print("===attn_out:{}".format(out.shape.as_list()))

        #-------spatial---------------
        s = model_input
        attn_spa = AttnLSTM_spa(1024)
        out_spa = attn_spa.forward(s)
        print("out_spatial:{}".format(out_spa.shape.as_list()))

        attn_spa2 = AttnLSTM_spa(1024,scope='attn_spa2')
        out_spa = attn_spa2.forward(out_spa)

        #x3
        attn_spa3 = AttnLSTM_spa(1024, scope='attn_spa3')
        out_spa = attn_spa3.forward(out_spa)
        #x4
        attn_spa4 = AttnLSTM_spa(1024, scope='attn_spa4')
        out_spa = attn_spa4.forward(out_spa)
        #fusion+

        # arf = tf.get_variable("arf", [1], initializer=tf.constant_initializer(0.0))
        # bta = tf.get_variable("bta", [1], initializer=tf.constant_initializer(0.0))
        # arf = tf.sigmoid(arf)*2
        # bta = tf.sigmoid(bta)*2
        # arf = tf.get_variable("arf", [1,2], initializer=tf.constant_initializer(0.0))
        # arf = tf.nn.softmax(arf)

        #out = tf.add(out*arf,out_spa*bta)
        #out = tf.add(out * arf[0][0], out_spa * arf[0][1])
        #print("out1:{}".format(out.shape.as_list()))
        out = tf.add(out * 0.9, out_spa * 0.1)

        #-----------------------双向差分运动编码--------------------------------
        #双向差分，单向编码
        #单向差分，双向编码
        # [None,40,1024]
        m_out = out
        forward_motion = m_out[:,1:40,:] - m_out[:, 0:39, :]
        back_motion = m_out[:, 0:39, :] - m_out[:,1:40,:]
        #反转
        #bw_motion = tf.reverse(back_motion,axis=[1])
        bw_motion = back_motion

        fm_w = tf.get_variable("fmw", [1], initializer=tf.constant_initializer(0.0))
        bw_w = tf.get_variable("bww", [1], initializer=tf.constant_initializer(0.0))
        fm_w = tf.sigmoid(fm_w)
        bw_w = tf.sigmoid(bw_w)

        # fm_bw_w = tf.get_variable("arf", [1,2], initializer=tf.constant_initializer(0.0))
        # fm_bw_w = tf.nn.softmax(fm_bw_w)

        #print("==model_input:{}".format(model_input.shape.as_list()))
        #model_input:[None,40,1024]
        with tf.variable_scope("RNN"):
            #src_input:model_input->out
            outputs, state = tf.nn.dynamic_rnn(stacked_lstm, out,
                                               sequence_length=num_frames,
                                               dtype=tf.float32)
        with tf.variable_scope("RNN1"):
            fm_outputs, fm_state = tf.nn.dynamic_rnn(forward_motion_lstm, forward_motion,
                                                 sequence_length=num_frames,
                                                 dtype=tf.float32)

        with tf.variable_scope("RNN2"):
            bk_outputs, bk_state = tf.nn.dynamic_rnn(back_motion_lstm, bw_motion,
                                                 sequence_length=num_frames,
                                                 dtype=tf.float32)
        #print("==state:{}".format(state.shape.as_list()))
        #state:[None,4096]
        #print("vieo_level_models:{}".format(video_level_models))
        #分类器
        #-----------------------------imp 双向差分运动编码-----------------



        #forward_motion encoding
        #outputs:[None,40,1024] state:[None,4096]

        print("===========forward_motion:{} fm_state:{} fm_outputs:{}".format(forward_motion.shape.as_list(),
                                                                              fm_state.shape.as_list(),
                                                                              fm_outputs.shape.as_list()))


        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        #-----------------------------------------imp---------------------------
        #运动和时序信息的联合表示
        print("!!!!state:{} fm_state:{}".format(state.shape.as_list(),fm_state.shape.as_list()))
        final_state = state + fm_w*fm_state + bw_w*bk_state
        #final_state = state + fm_state*0.7 + bk_state*0.7
        #final_state = state + fm_bw_w[0][0] * fm_state + fm_bw_w[0][1] * bk_state

        #state->final_state
        return aggregated_model().create_model(
            model_input=final_state,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class GruModel(models.BaseModel):

    def create_model(self, model_input, vocab_size, num_frames, is_training=True, **unused_params):
        """Creates a model which uses a stack of GRUs to represent the video.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        gru_size = FLAGS.gru_cells
        number_of_layers = FLAGS.gru_layers
        backward = FLAGS.gru_backward
        random_frames = FLAGS.gru_random_sequence
        iterations = FLAGS.iterations

        if random_frames:
            num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
            model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                                   iterations)

        if backward:
            model_input = tf.reverse_sequence(model_input, num_frames, seq_axis=1)

        stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
            ], state_is_tuple=False)

        loss = 0.0
        with tf.variable_scope("RNN"):
            outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                               sequence_length=num_frames,
                                               dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)


class NetFVModelLF(models.BaseModel):
    """Creates a NetFV based model.
       It emulates a Gaussian Mixture Fisher Vector pooling operations

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.fv_cluster_size
        hidden1_size = hidden_size or FLAGS.fv_hidden_size
        relu = FLAGS.fv_relu
        gating = FLAGS.gating

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        print("model_input(Before):", model_input)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
            print("model_input(After):", model_input)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.summary.histogram("input_hist", reshaped_input)

        video_NetFV = NetFV(1024, max_frames, cluster_size, add_batch_norm, is_training)
        # audio_NetFV = NetFV(128, max_frames, cluster_size / 2, add_batch_norm, is_training)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        with tf.variable_scope("video_FV"):
            fv_video = video_NetFV.forward(reshaped_input[:, 0:1024])

        # with tf.variable_scope("audio_FV"):
        #     fv_audio = audio_NetFV.forward(reshaped_input[:, 1024:])
        #
        # fv = tf.concat([fv_video, fv_audio], 1)
        fv = tf.concat([fv_video], 1)

        fv_dim = fv.get_shape().as_list()[1]
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [fv_dim, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

        activation = tf.matmul(fv, hidden1_weights)

        if add_batch_norm and relu:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")
        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases

        if relu:
            activation = tf.nn.relu6(activation)

        if gating:
            gating_weights = tf.get_variable("gating_weights_2",
                                             [hidden1_size, hidden1_size],
                                             initializer=tf.random_normal_initializer(
                                                 stddev=1 / math.sqrt(hidden1_size)))

            gates = tf.matmul(activation, gating_weights)

            if add_batch_norm:
                gates = slim.batch_norm(
                    gates,
                    center=True,
                    scale=True,
                    is_training=is_training,
                    scope="gating_bn")
            else:
                gating_biases = tf.get_variable("gating_biases",
                                                [cluster_size],
                                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
                gates += gating_biases

            gates = tf.sigmoid(gates)

            activation = tf.multiply(activation, gates)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            is_training=is_training,
            **unused_params)
