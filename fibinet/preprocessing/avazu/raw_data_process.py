import json
import os

import pandas as pd

from fibinet.common.data_loader import DataLoader
from fibinet.preprocessing.base_process import BaseProcess


class RawDataProcess(BaseProcess):
    """
    1. concat train.txt and userid_profile.txt to make a full train.txt file
    2. process label to 0 and 1
    """
    
    def __init__(self, config_path, ):
        super(RawDataProcess, self).__init__(config_path)
        self.concat_path = self.config.get("base_info", {}).get("concat_path", None)
        self.target_config_path = "{dir}/{name}".format(dir=os.path.dirname(self.config_path),
                                                        name="config_concat.json")
        
        self._init()
        return
    
    def _init(self):
        DataLoader.validate_or_create_dir(self.concat_path)
        return
    
    def fit(self):
        return
    
    def transform(self, sep="\t", chunksize=10000):
        target_file = os.path.join(self.concat_path, "train.txt")
        if os.path.exists(target_file):
            os.remove(target_file)
        
        label_index = 1
        iterator = pd.read_csv(os.path.join(self.train_path, "train.txt"), sep=sep, index_col=None,
                               chunksize=chunksize, encoding="utf-8")
        for n, data_chunk in enumerate(iterator):
            print('RawDataProcess::transform: Size of uploaded chunk: %i instances, %i features' % data_chunk.shape)
            print("RawDataProcess::transform: chunk counter: {}".format(n))
            
            # Put "label" to the first column
            cols = list(data_chunk.columns)
            cols.insert(0, cols.pop(label_index))
            # Remove column of "id" and "hour"
            cols.pop(1)
            cols.pop(1)
            df_target = data_chunk.loc[:, cols]
            
            df_target.to_csv(os.path.join(self.concat_path, "train.txt"), sep='\t', header=None, index=False, mode="a")
        
        self._update_config()
        pass
    
    def _update_config(self):
        self.config["base_info"]["train_path"] = self.concat_path
        with open(self.target_config_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.config, json_file, ensure_ascii=False, indent=4)
        return True
