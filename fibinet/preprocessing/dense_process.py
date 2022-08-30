import json
import math
import os
from collections import OrderedDict

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from fibinet.common.data_loader import DataLoader
from fibinet.common.utils import Utils
from fibinet.preprocessing.base_process import BaseProcess


class DenseProcess(BaseProcess):
    """
    Preprocessing for dense fields in datasets
    """
    
    def __init__(self, config_path, **kwargs):
        super(DenseProcess, self).__init__(config_path=config_path, **kwargs)
        self.dense_path = self.config.get("base_info", {}).get("dense_path", None)
        self.transformer = None
        self.target_config_path = "{dir}/{name}".format(dir=os.path.dirname(self.config_path),
                                                        name="config_dense.json")
        
        self.min_max_dict = None
        self.min_max_scaler = None
        self.min_max_scalers = None
        
        self._init()
        return
    
    def _init(self):
        DataLoader.validate_or_create_dir(self.dense_path)
        return
    
    def fit(self, transformer):
        self.transformer = transformer
        return True
    
    def transform(self, sep="\t"):
        if not os.path.exists(self.dense_path):
            os.mkdir(self.dense_path)
        
        cnt_train = 0
        dense_feature_indexes = set(self.dense_feature_indexes)
        for file_path in self.train_files:
            fi = open(file_path, 'r', encoding="utf-8")
            fo = open(self.dense_path + os.path.basename(file_path), 'w', encoding="utf-8")
            for line in fi:
                if cnt_train % 10000 == 0:
                    print('DenseProcess::transform: cnt : %d' % cnt_train)
                entry = []
                items = line.strip('\n').split(sep=sep)
                for index, item in enumerate(items):
                    if index in dense_feature_indexes:
                        entry.append(str(self.transformer(item, index)))
                    else:
                        entry.append(item)
                fo.write(sep.join(entry) + '\n')
                cnt_train += 1
            fi.close()
            fo.close()
        
        self._update_config()
        return True
    
    def scale_multi_min_max(self, x, index=-1):
        """
        Mode 0: value = (x-min)/(max-min) Min-Max-Scale
        :param x:
        :param index:
        :return:
        """
        if not self.min_max_scalers:
            self.min_max_scalers = self._fit_min_max_scaler()
        scaler = self.min_max_scalers.get(index)
        x = Utils.get_float(x)
        transformed_values = scaler.transform(np.reshape(x, (-1, 1)))
        value = transformed_values[0][0]
        return value
    
    def _fit_min_max_scaler(self, sep="\t"):
        if not self.min_max_dict:
            self.min_max_dict = self._get_min_max_dict(sep=sep)
        
        min_max_scalers = OrderedDict()
        for key, value in self.min_max_dict.items():
            scaler = MinMaxScaler()
            scaler.fit(np.reshape(value, (-1, 1)))
            min_max_scalers[key] = scaler
        return min_max_scalers
    
    def scale_ln2(self, x, index=-1):
        """
        Mode 1: value = int[(Ln(x))^2]
        :param x:
        :param index:
        :return:
        """
        if isinstance(x, str):
            x = Utils.get_float(x)
        if x > 2:
            x = int(math.log(float(x)) ** 2)
        return x
    
    def scale_ln0(self, x, index=-1):
        """
        Mode 2: value = ln(1+x-min)
        :param x:
        :param index:
        :return:
        """
        if not self.min_max_dict:
            self.min_max_dict = self._get_min_max_dict()
        
        x = Utils.get_float(x)
        current_min = self.min_max_dict.get(index, [0, 0])[0]
        value = math.log(1.0 + (float(x) - current_min))
        return value
    
    def _get_min_max_dict(self, sep="\t"):
        min_max_dict = OrderedDict()
        for i in self.dense_feature_indexes:
            min_max_dict[i] = [np.inf, -np.inf]  # min/max
        cnt_train = 0
        for file_path in self.train_files:
            fi = open(file_path, 'r', encoding="utf-8")
            for line in fi:
                if cnt_train % 10000 == 0:
                    print('DenseProcess::fit: cnt : %d' % cnt_train)
                items = line.strip('\n').split(sep=sep)
                for index in self.dense_feature_indexes:
                    item = items[index]
                    need_update = False
                    current_min, current_max = min_max_dict.get(index)
                    item_value = Utils.get_float(item)
                    if item_value < current_min:
                        current_min = item_value
                        need_update = True
                    if item_value > current_max:
                        current_max = item_value
                        need_update = True
                    if need_update:
                        min_max_dict[index] = [current_min, current_max]
                
                cnt_train += 1
        return min_max_dict
    
    def _update_config(self):
        self.config["base_info"]["train_path"] = self.config["base_info"]["dense_path"]
        with open(self.target_config_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.config, json_file, ensure_ascii=False, indent=4)
        return True
