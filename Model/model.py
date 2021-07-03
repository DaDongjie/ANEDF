import tensorflow as tf
import os
import pickle
from Utils.utils import *
import numpy as np
from scipy.special import expit as sigmoid
from copy import deepcopy
from Dataset.dataset import jaccard


w_init = lambda:tf.random_normal_initializer(stddev=0.02)

class Model(object):

    def __init__(self, config):
        self.config = config
        self.net_shape = config['net_shape']
        self.att_shape = config['att_shape']
        self.net_att_shape = config['net_att_shape']

        self.net_input_dim = config['net_input_dim']
        self.att_input_dim = config['att_input_dim']
        self.net_att_input_dim = config['net_att_input_dim']

        self.is_init = config['is_init']  # True
        self.pretrain_params_path = config['pretrain_params_path']

        self.num_net_layers = len(self.net_shape)
        self.num_att_layers = len(self.att_shape)
        self.num_net_att_layers = len(self.net_att_shape)


        if self.is_init:  # True
            if os.path.isfile(self.pretrain_params_path):  # True 是文件
                with open(self.pretrain_params_path, 'rb') as handle:
                    self.W_init, self.b_init = pickle.load(handle)


    def forward_net(self, x, drop_prob, reuse=False):

        with tf.variable_scope('net_encoder', reuse=reuse) as scope:
            cur_input = x
            print("forward_net:",cur_input.get_shape())

            # ============encoder===========
            struct = self.net_shape
            for i in range(self.num_net_layers):
                name = 'net_encoder' + str(i)
                if self.is_init: ###
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init())
                if i < self.num_net_layers - 1:
                    cur_input = lrelu(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                print("forward_net:",cur_input.get_shape())

            net_H = cur_input

            # ====================decoder=============
            struct.reverse()
            cur_input = net_H
            for i in range(self.num_net_layers - 1):
                name = 'net_decoder' + str(i)
                if self.is_init: ###
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1], kernel_initializer=w_init())
                cur_input = lrelu(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print("forward_net:", cur_input.get_shape())

            name = 'net_decoder' + str(self.num_net_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input, units=self.net_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]))
            else:
                cur_input = tf.layers.dense(cur_input, units=self.net_input_dim, kernel_initializer=w_init())
            cur_input = tf.nn.sigmoid(cur_input)
            x_recon = cur_input
            print("forward_net:",cur_input.get_shape())

            self.net_shape.reverse()

        return net_H, x_recon

    def forward_att(self, x, drop_prob, reuse=False):

        with tf.variable_scope('att_encoder', reuse=reuse) as scope:
            cur_input = x
            print("forward_att:", cur_input.get_shape())

            # ============encoder===========
            struct = self.att_shape
            for i in range(self.num_att_layers):
                name = 'att_encoder' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init())
                if i < self.num_att_layers - 1:
                    cur_input = lrelu(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                print("forward_att:", cur_input.get_shape())

            att_H = cur_input

            # ====================decoder=============
            struct.reverse()
            cur_input = att_H
            for i in range(self.num_att_layers - 1):
                name = 'att_decoder' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1], kernel_initializer=w_init())
                cur_input = lrelu(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print("forward_att:",cur_input.get_shape())

            name = 'att_decoder' + str(self.num_att_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input, units=self.att_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]))
            else:
                cur_input = tf.layers.dense(cur_input, units=self.att_input_dim, kernel_initializer=w_init())
            # cur_input = tf.nn.sigmoid(cur_input)
            x_recon = cur_input
            print("forward_att:",cur_input.get_shape())

            self.att_shape.reverse()

        return att_H, x_recon


    def forward_net_att(self, x_z, drop_prob, reuse=False):

        with tf.variable_scope('net_att_encoder', reuse=reuse) as scope:
            cur_input = x_z
            print("forward_net_att:", cur_input.get_shape())

            # ============encoder===========
            struct = self.net_att_shape
            for i in range(self.num_net_att_layers):
                name = 'net_att_encoder' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i], kernel_initializer=w_init())
                if i < self.num_net_att_layers - 1:
                    cur_input = lrelu(cur_input)
                    cur_input = tf.layers.dropout(cur_input, drop_prob)
                print("forward_net_att:", cur_input.get_shape())

            net_att_H = cur_input

            # ====================decoder=============
            struct.reverse()
            cur_input = net_att_H
            for i in range(self.num_net_att_layers - 1):
                name = 'net_att_decoder' + str(i)
                if self.is_init:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1],
                                                kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                                bias_initializer=tf.constant_initializer(self.b_init[name]))
                else:
                    cur_input = tf.layers.dense(cur_input, units=struct[i + 1], kernel_initializer=w_init())
                cur_input = lrelu(cur_input)
                cur_input = tf.layers.dropout(cur_input, drop_prob)
                print("forward_net_att:", cur_input.get_shape())

            name = 'net_att_decoder' + str(self.num_net_att_layers - 1)
            if self.is_init:
                cur_input = tf.layers.dense(cur_input, units=self.net_att_input_dim,
                                            kernel_initializer=tf.constant_initializer(self.W_init[name]),
                                            bias_initializer=tf.constant_initializer(self.b_init[name]))
            else:
                cur_input = tf.layers.dense(cur_input, units=self.net_att_input_dim, kernel_initializer=w_init())
            cur_input = tf.nn.sigmoid(cur_input)
            x_z_recon = cur_input
            print("forward_net_att:",cur_input.get_shape())

            self.net_att_shape.reverse()

        return net_att_H, x_z_recon


