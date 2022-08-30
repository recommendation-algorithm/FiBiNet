from fibinet.preprocessing.avazu.raw_data_process import RawDataProcess
from fibinet.preprocessing.kfold_process import KFoldProcess
from fibinet.preprocessing.sparse_process import SparseProcess


def main():
    chunksize = 10000000
    # Process raw data to one file
    raw_data = RawDataProcess(config_path="./config/avazu/config_template.json")
    raw_data.fit()
    raw_data.transform(sep=",", chunksize=chunksize)
    
    # Preprocessing categorical fields：Fill null and LowFreq value with index-base, and encode label
    sparse = SparseProcess(config_path=raw_data.target_config_path)
    sparse.fit(min_occurrences=4, index_base=0)
    sparse.transform()
    
    # K-Fold
    kfold = KFoldProcess(config_path=sparse.target_config_path)
    kfold.fit()
    kfold.transform(chunksize=chunksize)
    return


if __name__ == "__main__":
    main()
