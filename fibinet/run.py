import argparse
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from fibinet.common.constants import Constants
from fibinet.common.data_loader import DataLoader
from fibinet.common.utils import Utils
from fibinet.model.batch_generator import BatchGenerator
from fibinet.model.components.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from fibinet.model.fibinet_model import FiBiNetModel


# tf.enable_eager_execution()


class FiBiNetRunner(object):
    """
    train & test FiBiNetModel with supported args
    """
    CHECKPOINT_TEMPLATE = "cp-{epoch:04d}.ckpt"
    CHECKPOINT_RE_TEMPLATE = "^cp-(.*).ckpt"
    
    def __init__(self):
        # Get args
        self.args = self.parse_args()
        self._update_parameters()
        print("Args: ", self.args.__dict__)
        # Load config for config_path
        self.config = DataLoader.load_config_dict(config_path=self.args.config_path)
        self.model_config = self.config.get("model", {})
        # Load dataset feature info
        self._load_feature_info()
        # Get input/output files
        self._get_input_output_files()
        # Create dirs
        self._create_dirs()
        return
    
    def _update_parameters(self):
        if self.args.version == "v1":
            parameters = {
                "sparse_embedding_norm_type": "none",
                "dense_embedding_norm_type": "none",
                "senet_squeeze_mode": "mean",
                "senet_squeeze_group_num": 1,
                "senet_excitation_mode": "vector",
                "senet_activation": "relu",
                "senet_use_skip_connection": False,
                "senet_reweight_norm_type": "none",
                "origin_bilinear_type": "all",
                "origin_bilinear_dnn_units": [],
                "origin_bilinear_dnn_activation": "linear",
                "senet_bilinear_type": "all",
                "enable_linear": True
            }
            self.args.__dict__.update(parameters)
        elif self.args.version == "++":
            parameters = {
                "sparse_embedding_norm_type": "bn",
                "dense_embedding_norm_type": "layer_norm",
                "senet_squeeze_mode": "group_mean_max",
                "senet_squeeze_group_num": 2,
                "senet_excitation_mode": "bit",
                "senet_activation": "none",
                "senet_use_skip_connection": True,
                "senet_reweight_norm_type": "ln",
                "origin_bilinear_type": "all_ip",
                "origin_bilinear_dnn_units": [50],
                "origin_bilinear_dnn_activation": "linear",
                "senet_bilinear_type": "none",
                "enable_linear": False,
            }
            self.args.__dict__.update(parameters)
        return
    
    def _load_feature_info(self):
        """
        Load feature info of a dataset from config
        :return:
        """
        self.features = []
        self.sparse_features = []
        self.dense_features = []
        self.varlen_features = []
        for feature in self.config["features"]:
            if feature["type"] == Constants.FEATURE_TYPE_SPARSE:
                sparse_feature = SparseFeat(name=feature["name"], dimension=feature["dimension"],
                                            use_hash=feature["use_hash"],
                                            dtype=feature["dtype"], embedding=feature["embedding"])
                self.sparse_features.append(sparse_feature)
                self.features.append(sparse_feature)
            elif feature["type"] == Constants.FEATURE_TYPE_DENSE:
                dense_feature = DenseFeat(name=feature["name"], dimension=feature.get("dimension", 1),
                                          dtype=feature["dtype"])
                self.dense_features.append(dense_feature)
                self.features.append(dense_feature)
            elif feature["type"] == Constants.FEATURE_TYPE_VARLENSPARSE:
                varlen_feature = VarLenSparseFeat(name=feature["name"], dimension=feature["dimension"],
                                                  maxlen=feature["maxlen"], combiner=feature["combiner"],
                                                  use_hash=feature["use_hash"],
                                                  dtype=feature["dtype"], embedding=feature["embedding"])
                self.varlen_features.append(varlen_feature)
                self.features.append(varlen_feature)
        return True
    
    def _get_input_output_files(self):
        train_paths = self.args.train_paths if self.args.train_paths else self.model_config["train_paths"]
        valid_paths = self.args.valid_paths if self.args.valid_paths else self.model_config["valid_paths"]
        test_paths = self.args.test_paths if self.args.test_paths else self.model_config["test_paths"]
        print("Train paths: ", self.model_config["data_prefix"], train_paths)
        print("Valid paths: ", self.model_config["data_prefix"], valid_paths)
        print("Test paths: ", self.model_config["data_prefix"], test_paths)
        self.train_files = DataLoader.get_files(self.model_config["data_prefix"], train_paths)
        self.valid_files = DataLoader.get_files(self.model_config["data_prefix"], valid_paths)
        self.test_files = DataLoader.get_files(self.model_config["data_prefix"], test_paths)
        self.train_results_file = os.path.join(self.model_config["results_prefix"],
                                               self.args.model_path,
                                               self.model_config["train_results_file"])
        self.test_results_file = os.path.join(self.model_config["results_prefix"],
                                              self.args.model_path,
                                              self.model_config["test_results_file"])
        return
    
    def _create_dirs(self):
        self.checkpoint_dir = os.path.join(self.model_config.get("results_prefix", "./data/model/"),
                                           self.args.model_path, "checkpoint")
        self.model_file_dir = os.path.join(self.model_config.get("model_file_path", "./data/model/"),
                                           self.args.model_path, "model")
        DataLoader.validate_or_create_dir(self.checkpoint_dir)
        DataLoader.validate_or_create_dir(self.model_file_dir)
        return
    
    def create_checkpoint_callback(self, save_weights_only=True, period=1):
        """
        Create callback function of checkpoint
        :return: callback
        """
        checkpoint_path = "{checkpoint_dir}/{name}".format(checkpoint_dir=self.checkpoint_dir,
                                                           name=self.CHECKPOINT_TEMPLATE)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=save_weights_only,
            period=period, )
        return cp_callback
    
    def create_earlystopping_callback(self, monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto',
                                      baseline=None, restore_best_weights=True):
        """
        Create early stopping callback
        :return: callback
        """
        es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience,
                                                       verbose=verbose, mode=mode, baseline=baseline,
                                                       restore_best_weights=restore_best_weights)
        return es_callback
    
    def restore_model_from_checkpoint(self, restore_epoch=-1):
        """
        Restore model weights from checkpoint created by restore_epoch
        Notice: these are only weights in checkpoint file, model structure should by created by create_model
        :param restore_epoch: from which checkpoint to load weights
        :return: model, latest_epoch
        """
        # Get checkpoint path
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
        latest_epoch = self._get_latest_epoch_from_checkpoint(latest_checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, self.CHECKPOINT_TEMPLATE.format(
            epoch=restore_epoch)) if 0 < restore_epoch <= latest_epoch else latest_checkpoint_path
        
        # create model and load weights from checkpoint
        model = self.create_model()
        model.load_weights(checkpoint_path)
        print("BaseModel::restore_model_from_checkpoint: restore model from checkpoint: ", checkpoint_path)
        return model, latest_epoch
    
    def _get_latest_epoch_from_checkpoint(self, latest_checkpoint):
        """
        Get latest epoch from checkpoint path
        :param latest_checkpoint:
        :return:
        """
        latest_epoch = 0
        regular = re.compile(self.CHECKPOINT_RE_TEMPLATE)
        try:
            checkpoint = os.path.basename(latest_checkpoint)
            match_result = regular.match(checkpoint)
            latest_epoch = int(match_result.group(1))
        except Exception as e:
            print(e)
        return latest_epoch
    
    def parse_args(self):
        parser = argparse.ArgumentParser()
        # 1. Run setup
        parser.add_argument('--version', type=str, default="++",
                            help="#version: version of fibinet model, support v1, ++ and custom")
        parser.add_argument('--config_path', type=str, default="./config/criteo/config_dense.json",
                            help="#config path: path of config file which includes info of dataset features")
        parser.add_argument('--train_paths', type=Utils.str2liststr,
                            default="part0,part1,part2,part3,part4,part5,part6,part7",
                            help='#train_paths: training directories split with comma')
        parser.add_argument('--valid_paths', type=Utils.str2liststr, default="part8",
                            help='#valid_paths: validation directories split with comma')
        parser.add_argument('--test_paths', type=Utils.str2liststr, default="part9",
                            help='#test_paths: testing directories split with comma')
        # 2. Model architecture
        # 2.1. Embeddings
        parser.add_argument('--embedding_size', type=int, default=10,
                            help='#embedding_size: feature embedding size')
        parser.add_argument('--embedding_l2_reg', type=float, default=0.0,
                            help='#embedding_l2_reg: L2 regularizer strength applied to embedding')
        parser.add_argument('--embedding_dropout', type=float, default=0.0,
                            help='#embedding_dropout: the probability of dropping out on embedding')
        parser.add_argument('--sparse_embedding_norm_type', type=str, default='bn',
                            help='#sparse_embedding_norm_type: str, support `none, bn')
        parser.add_argument('--dense_embedding_norm_type', type=str, default='layer_norm',
                            help='#dense_embedding_norm_type: str, support `none, layer_norm')
        parser.add_argument('--dense_embedding_share_params', type=Utils.str2bool, default=False,
                            help='#dense_embedding_share_params: whether sharing params among different fields')
        
        # 2.2. SENet
        parser.add_argument('--senet_squeeze_mode', type=str, default='group_mean_max',
                            help='#senet_squeeze_mode: mean, max, topk, and group')
        parser.add_argument('--senet_squeeze_group_num', type=int, default=2,
                            help='#senet_squeeze_group_num: worked only in group mode')
        parser.add_argument('--senet_squeeze_topk', type=int, default=1,
                            help='#senet_squeeze_topk: positive integer, topk value')
        parser.add_argument('--senet_reduction_ratio', type=float, default=3.0,
                            help='#senet_reduction_ratio: senet reduction ratio')
        parser.add_argument('--senet_excitation_mode', type=str, default="bit",
                            help='#senet_excitation_mode: str, support: none(=squeeze_mode), vector|group|bit')
        parser.add_argument('--senet_activation', type=str, default='none',
                            help='#senet_activation: activation function used in SENet Layer 2')
        parser.add_argument('--senet_use_skip_connection', type=Utils.str2bool, default=True,
                            help='#senet_use_skip_connection:  bool.')
        parser.add_argument('--senet_reweight_norm_type', type=str, default='ln',
                            help='#senet_reweight_norm_type: none, ln')
        
        # 2.3. Bilinear type
        parser.add_argument('--origin_bilinear_type', type=str, default='all_ip',
                            help='#origin_bilinear_type: bilinear type applied to original embeddings')
        parser.add_argument('--origin_bilinear_dnn_units', type=Utils.str2list, default=[50],
                            help='#origin_bilinear_dnn_units: list')
        parser.add_argument('--origin_bilinear_dnn_activation', type=str, default='linear',
                            help='#origin_bilinear_dnn_activation: Activation function to use in DNN')
        parser.add_argument('--senet_bilinear_type', type=str, default='none',
                            help='#senet_bilinear_type: bilinear type applied to senet embeddings')
        
        # 2.4. DNN part
        parser.add_argument('--dnn_hidden_units', type=Utils.str2list, default=[400, 400, 400],
                            help='#dnn_hidden_units: layer number and units in each layer of DNN')
        parser.add_argument('--dnn_activation', type=str, default='relu',
                            help='#dnn_activation: activation function used in DNN')
        parser.add_argument('--dnn_l2_reg', type=float, default=0.0,
                            help='#dnn_l2_reg: L2 regularizer strength applied to DNN')
        parser.add_argument('--dnn_use_bn', type=Utils.str2bool, default=False,
                            help='#dnn_use_bn: whether to use BatchNormalization before activation or not in DNN')
        parser.add_argument('--dnn_dropout', type=float, default=0.0,
                            help='#dnn_dropout: the probability of dropping out on each layer of DNN')
        
        # 2.5. Linear part
        parser.add_argument('--enable_linear', type=Utils.str2bool, default=False,
                            help='#enable_linear:  bool. Whether use linear part in the model')
        parser.add_argument('--linear_l2_reg', type=float, default=0.0,
                            help='#linear_l2_reg: L2 regularizer strength applied to linear')
        
        # 3. Train/Valid/Test setup
        parser.add_argument('--seed', type=int, default=1024, help='#seed: integer ,to use as random seed.')
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--batch_size', type=int, default=1024)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--init_std', type=float, default=0.01)
        parser.add_argument('--verbose', type=int, default=1)
        parser.add_argument("--mode", type=str, default="train", help="support: train, retrain, test")
        parser.add_argument('--restore_epochs', type=Utils.str2list, default=[],
                            help="restore weights from checkpoint, format like np.arange(), eg. [1, 5, 1]")
        parser.add_argument("--early_stopping", type=Utils.str2bool, default=True, help="enable early stopping")
        parser.add_argument("--model_path", type=str, default="fibinet", help="model_path, to avoid being covered")
        return parser.parse_args()
    
    def create_model(self):
        """
        Create FiBiNet model
        :return: instance of FiBiNet model: tf.keras.Model
        """
        fibinet = FiBiNetModel(params=self.args.__dict__,
                               feature_columns=self.features,
                               embedding_size=self.args.embedding_size,
                               embedding_l2_reg=self.args.embedding_l2_reg,
                               embedding_dropout=self.args.embedding_dropout,
                               sparse_embedding_norm_type=self.args.sparse_embedding_norm_type,
                               dense_embedding_norm_type=self.args.dense_embedding_norm_type,
                               dense_embedding_share_params=self.args.dense_embedding_share_params,
                               senet_squeeze_mode=self.args.senet_squeeze_mode,
                               senet_squeeze_group_num=self.args.senet_squeeze_group_num,
                               senet_squeeze_topk=self.args.senet_squeeze_topk,
                               senet_reduction_ratio=self.args.senet_reduction_ratio,
                               senet_excitation_mode=self.args.senet_excitation_mode,
                               senet_activation=self.args.senet_activation,
                               senet_use_skip_connection=self.args.senet_use_skip_connection,
                               senet_reweight_norm_type=self.args.senet_reweight_norm_type,
                               origin_bilinear_type=self.args.origin_bilinear_type,
                               origin_bilinear_dnn_units=self.args.origin_bilinear_dnn_units,
                               origin_bilinear_dnn_activation=self.args.origin_bilinear_dnn_activation,
                               senet_bilinear_type=self.args.senet_bilinear_type,
                               dnn_hidden_units=self.args.dnn_hidden_units,
                               dnn_activation=self.args.dnn_activation,
                               dnn_l2_reg=self.args.dnn_l2_reg,
                               dnn_use_bn=self.args.dnn_use_bn,
                               dnn_dropout=self.args.dnn_dropout,
                               enable_linear=self.args.enable_linear,
                               linear_l2_reg=self.args.linear_l2_reg,
                               init_std=self.args.init_std,
                               seed=self.args.seed, )
        model = fibinet.get_model()
        # optimizer & loss & metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate, beta_1=0.9, beta_2=0.999,
                                             epsilon=1e-8)
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = ["AUC", "binary_crossentropy"]
        model.compile(optimizer, loss, metrics=metrics)
        # model.run_eagerly = True
        # Print Info
        model.summary()
        tf.keras.utils.plot_model(model, os.path.join(self.model_file_dir, "fibinet.png"), show_shapes=True,
                                  show_layer_names=True)
        return model
    
    def run(self):
        if self.args.mode in (Constants.MODE_TRAIN, Constants.MODE_RETRAIN):
            self.train_model()
            self.test_model()
        elif self.args.mode == Constants.MODE_TEST:
            self.test_model()
        return True
    
    def train_model(self):
        """
        Train FiBiNET model
        :return: history
        """
        if self.args.mode == Constants.MODE_RETRAIN:
            model, latest_epoch = self.restore_model_from_checkpoint()
        else:
            model = self.create_model()
            latest_epoch = 0
        callbacks = [self.create_checkpoint_callback(), self.create_earlystopping_callback(), ]
        
        # 1. Get data from generator (single process & thread)
        train_steps = BatchGenerator.get_txt_dataset_length(self.train_files, batch_size=self.args.batch_size,
                                                            drop_remainder=True)
        val_steps = BatchGenerator.get_txt_dataset_length(self.valid_files, batch_size=self.args.batch_size,
                                                          drop_remainder=False)
        train_generator = BatchGenerator.generate_arrays_from_file(self.train_files, batch_size=self.args.batch_size,
                                                                   drop_remainder=True, features=self.features,
                                                                   shuffle=True)
        val_generator = BatchGenerator.generate_arrays_from_file(self.valid_files, batch_size=self.args.batch_size,
                                                                 drop_remainder=False, features=self.features,
                                                                 shuffle=False)
        
        history = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=self.args.epochs,
                                      verbose=self.args.verbose, validation_data=val_generator,
                                      validation_steps=val_steps,
                                      callbacks=callbacks, max_queue_size=10, workers=1, use_multiprocessing=False,
                                      shuffle=False, initial_epoch=latest_epoch)
        self._save_train_results(latest_epoch, history)
        return history
    
    def _save_train_results(self, latest_epoch, history):
        df = pd.DataFrame(history.history)
        df.insert(0, "epoch", range(latest_epoch + 1, latest_epoch + len(df) + 1))
        if len(df) > 0:
            df.to_csv(self.train_results_file, sep="\t", float_format="%.5f", index=False, encoding="utf-8", mode="a")
        return
    
    def test_model(self):
        """
        Test FiBiNET model, support to test model from specific checkpoint
        :return:
        """
        restore_epochs = []
        if not isinstance(self.args.restore_epochs, list) or len(self.args.restore_epochs) == 0:
            restore_epochs = np.arange(1, self.args.epochs + 1)
        elif len(self.args.restore_epochs) == 1:
            restore_epochs = np.arange(1, self.args.restore_epochs[0])
        elif len(self.args.restore_epochs) == 2:
            restore_epochs = np.arange(self.args.restore_epochs[0], self.args.restore_epochs[1])
        elif len(self.args.restore_epochs) >= 3:
            restore_epochs = np.arange(self.args.restore_epochs[0], self.args.restore_epochs[1],
                                       self.args.restore_epochs[2])
        print("FiBiNetRunner::test_model: restore_epochs: {}".format(restore_epochs))
        for restore_epoch in restore_epochs:
            self.test_model_from_checkpoint(restore_epoch)
        return True
    
    def test_model_from_checkpoint(self, restore_epoch=-1):
        model, latest_epoch = self.restore_model_from_checkpoint(restore_epoch=restore_epoch)
        test_steps = BatchGenerator.get_dataset_length(self.test_files, batch_size=self.args.batch_size,
                                                       drop_remainder=False)
        test_generator = BatchGenerator.generate_arrays_from_file(self.test_files, batch_size=self.args.batch_size,
                                                                  features=self.features, drop_remainder=False,
                                                                  shuffle=False)
        pred_ans = model.evaluate_generator(test_generator, steps=test_steps, verbose=self.args.verbose)
        results_dict = dict(zip(model.metrics_names, pred_ans))
        print("FiBiNetRunner::test_model_from_checkpoint: Epoch {} Evaluation results: {}".format(restore_epoch,
                                                                                                  results_dict))
        self._save_test_results(restore_epoch, results_dict)
        return
    
    def _save_test_results(self, restore_epoch, results_dict):
        df = pd.DataFrame(columns=results_dict.keys())
        df.loc[0] = list(results_dict.values())
        df.insert(0, "epoch", "{}".format(restore_epoch))
        if len(df) > 0:
            df.to_csv(self.test_results_file, sep="\t", float_format="%.5f", index=False, encoding="utf-8", mode="a")
        return


if __name__ == "__main__":
    runner = FiBiNetRunner()
    runner.run()
    pass
