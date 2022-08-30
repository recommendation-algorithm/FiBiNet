#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import ast

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class Utils:
    @staticmethod
    def count_parameters(model):
        """
        Counting model parameters
        :return: total_count, trainable_count, non_trainable_count
        """
        trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_count = trainable_count + non_trainable_count
        
        print('Total params: {:,}'.format(total_count))
        print('Trainable params: {:,}'.format(trainable_count))
        print('Non-trainable params: {:,}'.format(non_trainable_count))
        
        return total_count, trainable_count, non_trainable_count
    
    @staticmethod
    def str2list(v):
        try:
            v = v.split(',')
            v = [int(_.strip('[]')) for _ in v]
        except:
            v = []
        return v
    
    @staticmethod
    def str_to_type(v):
        try:
            v = ast.literal_eval(v)
        except:
            v = []
        return v
    
    @staticmethod
    def str2bool(v):
        if v.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif v.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    @staticmethod
    def get_float(value, default_value=0.0):
        try:
            return float(value)
        except:
            return default_value
    
    @staticmethod
    def str2liststr(v):
        v = v.split(",")
        v = [s.strip() for s in v]
        return v
    
    @staticmethod
    def get_upper_triangular_indices(n):
        indices = []
        for i in range(0, n):
            for j in range(i + 1, n):
                indices.append([i, j])
        return indices
    
    @staticmethod
    def concat_func(inputs, axis=-1):
        if len(inputs) == 1:
            return inputs[0]
        else:
            return tf.keras.layers.Concatenate(axis=axis)(inputs)
