import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .base_process import BaseProcess
from ..common.data_loader import DataLoader


class KFoldProcess(BaseProcess):
    """
    Split dataset with K-Fold
    """
    
    def __init__(self, config_path, **kwargs):
        super(KFoldProcess, self).__init__(config_path=config_path, **kwargs)
        self.k = self.config["base_info"]["k_fold"]
        self.k_fold_path = self.config["base_info"]["k_fold_path"]
        self.random_seed = self.config["base_info"]["random_seed"]
        self._init()
        return
    
    def _init(self):
        DataLoader.validate_or_create_dir(self.k_fold_path)
        return
    
    def fit(self):
        DataLoader.rmdirs(self.k_fold_path)
        return True
    
    def transform(self, sep="\t", chunksize=10000, shuffle=True):
        counter = 0
        feature_indexes = self.sparse_feature_indexes + self.dense_feature_indexes + self.varlen_feature_target_indexes
        feature_indexes.sort()
        for file_path in self.train_files:
            print("KFoldProcess::transform: Processing file {} ...".format(file_path))
            filename = os.path.basename(file_path)
            iterator = pd.read_csv(file_path, sep=sep, header=None, index_col=None, chunksize=chunksize,
                                   encoding="utf-8", dtype=None)
            for n, data_chunk in enumerate(iterator):
                print('KFoldProcess::transform: Size of uploaded chunk: %i instances, %i features' % data_chunk.shape)
                counter += 1
                print("KFoldProcess::transform: chunk counter: {}".format(counter))
                label_and_feature = data_chunk.values
                train_y = label_and_feature[:, self.label_index]
                train_x = label_and_feature[:, feature_indexes]
                print("KFoldProcess::transform: shape of train_x", train_x.shape)
                folds = list(StratifiedKFold(n_splits=self.k, shuffle=shuffle,
                                             random_state=self.random_seed).split(train_x, train_y))
                print("KFoldProcess::transform: fold num: %d" % (len(folds)))
                self._save_fold_data(folds, data_chunk, filename, sep)
                print("KFoldProcess::transform: save train_data done")
        return True
    
    def _save_fold_data(self, folds, train_data, filename, sep="\t"):
        for i in range(len(folds)):
            train_id, valid_id = folds[i]
            print("KFoldProcess::transform: now part %d" % i)
            file_path = os.path.join(self.k_fold_path, "part" + str(i) + "/")
            DataLoader.validate_or_create_dir(file_path)
            abs_filename = os.path.join(file_path, filename)
            with open(abs_filename, "a", encoding="utf-8") as f:
                for row_ix in valid_id:
                    items = [str(train_data.iat[row_ix, column_ix]) for column_ix in train_data.columns]
                    f.write(sep.join(items) + "\n")
        return True
