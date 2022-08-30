#!/usr/bin/env python
# -*- coding:utf-8 -*-


class Constants(object):
    """
    Constants
    """
    LOGGER_DEFAULT = "default"
    
    # feature type
    FEATURE_TYPE_LABEL = "label"
    FEATURE_TYPE_SPARSE = "sparse"
    FEATURE_TYPE_DENSE = "dense"
    FEATURE_TYPE_VARLENSPARSE = "VarLenSparse"
    
    # mode
    MODE_TRAIN = "train"
    MODE_RETRAIN = "retrain"
    MODE_TEST = "test"
    
    # date
    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
