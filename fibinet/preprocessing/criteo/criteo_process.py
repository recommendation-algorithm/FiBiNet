from fibinet.preprocessing.dense_process import DenseProcess
from fibinet.preprocessing.kfold_process import KFoldProcess
from fibinet.preprocessing.sparse_process import SparseProcess


def main(config):
    # Preprocessing categorical fields：Fill null and LowFreq value with index-base, and encode label
    sparse = SparseProcess(config_path="./config/criteo/config_template.json")
    sparse.fit(min_occurrences=10)
    sparse.transform()
    
    # Preprocessing dense fields：Fill null with 0, and do scale-transform
    config_path = sparse.target_config_path
    if config.get("enable_dense_scale", True):
        dense = DenseProcess(config_path=config_path)
        dense_transformer = config.get("dense_transformer", "scale_multi_min_max")
        if dense_transformer == "scale_ln2":
            dense.fit(dense.scale_ln2)
        elif dense_transformer == "scale_multi_min_max":
            dense.fit(dense.scale_multi_min_max)
        elif dense_transformer == "scale_ln0":
            dense.fit(dense.scale_ln0)
        else:
            raise Exception("Don't support parameter dense_transformer: {}".format(dense_transformer))
        dense.transform()
        config_path = dense.target_config_path
    
    # K-Fold
    # config_path = "./config/criteo/config_dense.json"
    kfold = KFoldProcess(config_path=config_path)
    kfold.fit()
    kfold.transform()
    return


if __name__ == "__main__":
    config = {
        "enable_dense_scale": True,
        "dense_transformer": "scale_multi_min_max",
    }
    main(config)
    pass
