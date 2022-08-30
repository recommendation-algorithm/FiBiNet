#!/usr/bin/env python
# -*- coding:utf-8 -*-
import collections
import json
import os

import numpy as np
import pandas as pd


class DataLoader(object):
    @staticmethod
    def load_config_dict(config_path):
        with open(config_path) as config_file:
            # for keep order
            config = json.loads(config_file.read(), object_pairs_hook=collections.OrderedDict)
        return config
    
    @staticmethod
    def get_files(prefix, paths):
        abs_paths = [os.path.join(prefix, path) for path in paths]
        files = DataLoader.list_multi_dir_files(abs_paths)
        return files
    
    @staticmethod
    def smart_load_data(path):
        names = os.path.splitext(path)
        if len(names) == 1:
            return False
        ext = names[1]
        if ext == ".json":
            return DataLoader.load_data_json(path)
        elif ext == ".txt":
            return DataLoader.load_data_txt(path)
        elif ext == ".csv":
            return DataLoader.load_data_txt(path, sep=",")
        elif ext == ".npy":
            return DataLoader.load_data_npy(path)
        return False
    
    @staticmethod
    def load_data_txt(path, sep="\t", names=None):
        df = pd.read_csv(path, names=names, sep=sep, index_col=False, header=None)
        return df.values
    
    @staticmethod
    def load_data_txt_as_df(path, sep="\t", names=None, usecols=None):
        df = pd.read_csv(path, names=names, sep=sep, index_col=False, header=None, usecols=usecols)
        return df
    
    @staticmethod
    def load_data_npy(path):
        data = np.load(path)
        return data
    
    @staticmethod
    def load_data_json(path):
        with open(path) as f:
            lines = f.read()
            config = json.loads(lines, object_pairs_hook=collections.OrderedDict)
        return config
    
    @staticmethod
    def list_dir_files(path):
        files = []
        for filename in os.listdir(path):
            abs_path = os.path.abspath(os.path.join(path, filename))
            if os.path.isfile(abs_path) and not os.path.basename(abs_path).startswith("."):
                files.append(abs_path)
        files.sort()
        return files
    
    @staticmethod
    def rmdirs(path):
        if not os.path.exists(path):
            return True
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        return True
    
    @staticmethod
    def list_multi_dir_files(paths):
        files = []
        for path in paths:
            files.extend(DataLoader.list_dir_files(path))
        return files
    
    @staticmethod
    def validate_or_create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return True
    
    @staticmethod
    def get_file_len(filename):
        cnt = 0
        with open(filename) as f:
            for line in f:
                line = line.strip("(\n| )")
                if line:
                    cnt += 1
        return cnt
