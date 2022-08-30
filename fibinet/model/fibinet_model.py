import copy

import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Add, Flatten, BatchNormalization

from fibinet.common.utils import Utils
from fibinet.model.components.inputs import build_input_features, get_linear_logit, input_from_feature_columns
from fibinet.model.components.layers import BilinearInteraction, DNNLayer, SENETLayer
from fibinet.model.components.layers import DenseEmbeddingLayer
from fibinet.model.components.layers import PredictionLayer
from fibinet.model.components.layers import SimpleLayerNormalization


class FiBiNetModel(object):
    def __init__(self, params, feature_columns, embedding_size=10, embedding_l2_reg=0.0, embedding_dropout=0.0,
                 sparse_embedding_norm_type="bn", dense_embedding_norm_type="layer_norm",
                 dense_embedding_share_params=False, senet_squeeze_mode="group_mean_max",
                 senet_squeeze_group_num=2, senet_squeeze_topk=1, senet_reduction_ratio=3.0,
                 senet_excitation_mode="bit", senet_activation="none", senet_use_skip_connection=True,
                 senet_reweight_norm_type="ln", origin_bilinear_type='all_ip', origin_bilinear_dnn_units=(50,),
                 origin_bilinear_dnn_activation="linear", senet_bilinear_type='none',
                 dnn_hidden_units=(400, 400, 400), dnn_l2_reg=0.0, dnn_use_bn=False, dnn_dropout=0.0,
                 dnn_activation='relu', enable_linear=False, linear_l2_reg=0.0, init_std=0.01, seed=1024,
                 task='binary'):
        """Instantiates the Feature Importance and Bilinear feature Interaction NETwork architecture.
        
        :param params: dict, contains all command line parameters
        :param feature_columns: An iterable containing all the features used by the model.
        :param embedding_size: positive integer,sparse feature embedding_size
        :param embedding_l2_reg: float. L2 regularize strength applied to embedding vector
        :param embedding_dropout: float, dropout probability applied to embedding vector
        :param sparse_embedding_norm_type: str, norm type for sparse fields, supports none,bn
        :param dense_embedding_norm_type: str, norm type for dense fields, supports none,layer_norm
        :param dense_embedding_share_params: bool, whether sharing parameters when using layer norm
        :param senet_reduction_ratio: integer in [1,inf), reduction ratio used in SENET Layer, default to 3
        :param senet_squeeze_mode: str, squeeze mode in SENet, support mean, max, min, topk
        :param senet_squeeze_topk: int, topk value when squeeze mode is topk
        :param senet_activation: str, activation for senet excitation, supports none, relu
        :param senet_use_skip_connection: bool, whether use skip connection in re-weights of SENet
        :param origin_bilinear_type: str, bilinear function type used in Bilinear Interaction Layer for the original
            embeddings, can be all, each, interaction
        :param senet_bilinear_type: str, bilinear function type used in Bilinear Interaction Layer for the senet
            embeddings, can be all, each, interaction
        :param dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer
            of DNN
        :param dnn_l2_reg: float. L2 regularize strength applied to DNN
        :param dnn_use_bn: bool, whether to use batch norm in DNN
        :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
        :param dnn_activation: Activation function to use in DNN
        :param linear_l2_reg: float. L2 regularize strength applied to wide part
        :param init_std: float,to use as the initialize std of embedding vector
        :param seed: integer ,to use as random seed.
        :param task: str, ``"binary"`` for  binary log_loss or  ``"regression"`` for regression loss
        :return: A FiBiNetModel instance
        """
        super(FiBiNetModel, self).__init__()
        tf.compat.v1.set_random_seed(seed=seed)
        
        self.params = copy.deepcopy(params)
        self.feature_columns = feature_columns
        self.field_size = len(feature_columns)
        
        self.embedding_size = embedding_size
        self.embedding_l2_reg = embedding_l2_reg
        self.embedding_dropout = embedding_dropout
        self.sparse_embedding_norm_type = sparse_embedding_norm_type
        self.dense_embedding_norm_type = dense_embedding_norm_type
        self.dense_embedding_share_params = dense_embedding_share_params
        
        self.senet_squeeze_mode = senet_squeeze_mode
        self.senet_squeeze_group_num = senet_squeeze_group_num
        self.senet_squeeze_topk = senet_squeeze_topk
        self.senet_reduction_ratio = senet_reduction_ratio
        self.senet_excitation_mode = senet_excitation_mode
        self.senet_activation = senet_activation
        self.senet_use_skip_connection = senet_use_skip_connection
        self.senet_reweight_norm_type = senet_reweight_norm_type
        
        self.origin_bilinear_type = origin_bilinear_type
        self.origin_bilinear_dnn_units = origin_bilinear_dnn_units
        self.origin_bilinear_dnn_activation = origin_bilinear_dnn_activation
        self.senet_bilinear_type = senet_bilinear_type
        
        self.dnn_hidden_units = dnn_hidden_units
        self.dnn_activation = dnn_activation
        self.dnn_l2_reg = dnn_l2_reg
        self.dnn_use_bn = dnn_use_bn
        self.dnn_dropout = dnn_dropout
        
        self.enable_linear = enable_linear
        self.linear_l2_reg = linear_l2_reg
        
        self.init_std = init_std
        self.seed = seed
        self.task = task
        
        self.features = None
        self.inputs_list = None
        self.embeddings = None
        self.senet_embeddings = None
        self.origin_bilinear_out = None
        self.senet_bilinear_out = None
        self.output = None
        return
    
    def get_model(self):
        # Inputs
        self.features, self.inputs_list = self.get_inputs()
        
        # Embeddings
        self.embeddings = self.get_embeddings()
        self.senet_embeddings = self.get_senet_embeddings()
        
        # Bilinear interaction
        self.origin_bilinear_out = BilinearInteraction(
            bilinear_type=self.origin_bilinear_type, dnn_units=self.origin_bilinear_dnn_units,
            dnn_activation=self.origin_bilinear_dnn_activation, seed=self.seed)(self.embeddings)  # [batch, 1, dim]
        self.senet_bilinear_out = BilinearInteraction(bilinear_type=self.senet_bilinear_type, seed=self.seed)(
            self.senet_embeddings)
        
        # DNN part
        dnn_input = Utils.concat_func([self.origin_bilinear_out, self.senet_bilinear_out])
        flatten_dnn_input = Flatten()(dnn_input)
        dnn_out = DNNLayer(self.dnn_hidden_units, self.dnn_activation, self.dnn_l2_reg, self.dnn_dropout,
                           self.dnn_use_bn, self.seed)(flatten_dnn_input)
        dnn_logit = Dense(1, use_bias=False, activation=None)(dnn_out)
        
        # Model
        if self.enable_linear:
            # Linear part
            linear_logit = get_linear_logit(self.features, self.feature_columns, l2_reg=self.linear_l2_reg,
                                            init_std=self.init_std, use_bias=False,
                                            seed=self.seed, prefix='linear')
            final_logit = Add()([linear_logit, dnn_logit])
        else:
            final_logit = dnn_logit
        self.output = PredictionLayer(self.task, use_bias=True)(final_logit)
        model = tf.keras.models.Model(inputs=self.inputs_list, outputs=self.output)
        return model
    
    def get_inputs(self):
        features = build_input_features(self.feature_columns)
        inputs_list = list(features.values())
        return features, inputs_list
    
    def get_embeddings(self, name_prefix=""):
        """
        Get sparse+dense feature embeddings
        :return:
        """
        # 1. sparse embedding
        sparse_embedding_list, dense_value_list = input_from_feature_columns(self.features, self.feature_columns,
                                                                             self.embedding_size,
                                                                             self.embedding_l2_reg,
                                                                             self.init_std,
                                                                             self.seed,
                                                                             prefix=name_prefix)
        sparse_embeddings = Utils.concat_func(sparse_embedding_list, axis=1)
        
        if self.sparse_embedding_norm_type in {"bn"}:
            sparse_embedding_norm_layer = BatchNormalization(axis=-1)
            sparse_embeddings = sparse_embedding_norm_layer(sparse_embeddings)
        
        # 2. dense embedding
        dense_embeddings = None
        if len(dense_value_list) > 0:
            dense_values = tf.stack(dense_value_list, axis=1)
            dense_embeddings = DenseEmbeddingLayer(embedding_size=self.embedding_size, init_std=self.init_std,
                                                   embedding_l2_reg=self.embedding_l2_reg, seed=self.seed)(dense_values)
            # DenseEmbeddingNorm
            if self.dense_embedding_norm_type in {"layer_norm"}:
                dense_embedding_norm_layer = SimpleLayerNormalization(self.dense_embedding_share_params)
                dense_embeddings = dense_embedding_norm_layer(dense_embeddings)
        
        # 3. get the whole embeddings
        if len(dense_value_list) > 0:
            embeddings = Utils.concat_func([sparse_embeddings, dense_embeddings], axis=1)
        else:
            embeddings = sparse_embeddings
        return embeddings
    
    def get_senet_embeddings(self):
        outputs = SENETLayer(senet_squeeze_mode=self.senet_squeeze_mode,
                             senet_squeeze_group_num=self.senet_squeeze_group_num,
                             senet_squeeze_topk=self.senet_squeeze_topk,
                             senet_reduction_ratio=self.senet_reduction_ratio,
                             senet_excitation_mode=self.senet_excitation_mode,
                             senet_activation=self.senet_activation,
                             senet_use_skip_connection=self.senet_use_skip_connection,
                             senet_reweight_norm_type=self.senet_reweight_norm_type,
                             seed=self.seed)(self.embeddings)
        return outputs
