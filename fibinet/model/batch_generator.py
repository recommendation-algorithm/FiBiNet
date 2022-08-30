import numpy as np

from fibinet.common.data_loader import DataLoader
from fibinet.model.components.inputs import VarLenSparseFeat


class BatchGenerator(object):
    @staticmethod
    def get_batch(samples, batch_size, index, features):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(samples) else len(samples)
        
        batch = samples[start:end].copy()
        x = []
        y = [batch[:, 0]]
        column_index = 1
        for feature in features:
            if isinstance(feature, VarLenSparseFeat):
                x.append(batch[:, column_index:(column_index + feature.maxlen)])
                column_index += feature.maxlen
            else:
                x.append(batch[:, column_index])
                column_index += 1
        return x, y  # labels, samples
    
    @staticmethod
    def generate_arrays_from_file(paths, batch_size=32, drop_remainder=True, features=None, shuffle=True):
        if not features:
            features = {}
        pre_path = ""
        samples = None
        while True:
            for path in paths:
                if pre_path != path:
                    samples = DataLoader.smart_load_data(path)
                    pre_path = path
                if len(samples) == 0:
                    continue
                if shuffle:
                    np.random.shuffle(samples)
                total_batch = int(len(samples) / batch_size) if drop_remainder else int(
                    (len(samples) - 1) / batch_size) + 1
                for index in range(total_batch):
                    batch_features, labels = BatchGenerator.get_batch(samples, batch_size, index, features)
                    yield batch_features, labels
    
    @staticmethod
    def get_dataset_length(paths, batch_size=32, drop_remainder=True):
        length = 0
        for path in paths:
            samples = DataLoader.smart_load_data(path)
            if len(samples) == 0:
                continue
            total_batch = int(len(samples) / batch_size) if drop_remainder else int((len(samples) - 1) / batch_size) + 1
            length += total_batch
        return length
    
    @staticmethod
    def get_txt_dataset_length(paths, batch_size=32, drop_remainder=True):
        length = 0
        for path in paths:
            file_length = DataLoader.get_file_len(path)
            total_batch = int(file_length / batch_size) if drop_remainder else int(
                (file_length - 1) / batch_size) + 1
            if total_batch == 0:
                continue
            length += total_batch
        return length
