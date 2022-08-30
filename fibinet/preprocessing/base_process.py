import logging

from fibinet.common.constants import Constants
from fibinet.common.data_loader import DataLoader


class BaseProcess(object):
    """
    Base class for dataset processing
    """
    
    def __init__(self, config_path, **kwargs):
        self.config_path = config_path
        self.config = DataLoader.load_config_dict(config_path=config_path)
        self.logger = logging.getLogger(Constants.LOGGER_DEFAULT)
        self.label_index, self.sparse_feature_indexes, self.dense_feature_indexes, self.varlen_feature_origin_indexes, \
        self.varlen_feature_target_indexes = self.get_feature_indexes()
        self.train_path = self.config.get("base_info", {}).get("train_path", None)
        self.train_files = DataLoader.list_dir_files(self.train_path)
        return
    
    def get_feature_indexes(self):
        label_index = 0
        sparse_feature_indexes = []
        dense_feature_indexes = []
        varlen_feature_origin_indexes = []
        varlen_feature_target_indexes = []
        features = [feature for feature in self.config.get("features", [])]
        if len(features) == 0:
            print("BaseProcess::get_feature_indexes: No features defined")
            exit(-1)
        index = 0
        for i, feature in enumerate(features):
            feature_type = feature.get("type", "")
            if feature_type == Constants.FEATURE_TYPE_LABEL:
                label_index = index
                index += 1
            elif feature_type == Constants.FEATURE_TYPE_SPARSE:
                sparse_feature_indexes.append(index)
                index += 1
            elif feature_type == Constants.FEATURE_TYPE_DENSE:
                dense_feature_indexes.append(index)
                index += 1
            elif feature_type == Constants.FEATURE_TYPE_VARLENSPARSE:
                varlen_feature_origin_indexes.append(i)
                for j in range(index, index + feature.get("maxlen", 1)):
                    varlen_feature_target_indexes.append(j)
                    index += 1
        return label_index, sparse_feature_indexes, dense_feature_indexes, varlen_feature_origin_indexes, \
               varlen_feature_target_indexes
