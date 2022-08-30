#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

from collections import OrderedDict, namedtuple

from tensorflow.python import GlorotNormal
from tensorflow.python.keras.initializers import RandomNormal, TruncatedNormal, RandomUniform
from tensorflow.python.keras.layers import Embedding, Input
from tensorflow.python.keras.regularizers import l2

from .layers import Hash, Linear
from fibinet.common.utils import Utils

class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()
    
    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()
    
    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)


class VarLenSparseFeat(namedtuple('VarLenFeat',
                                  ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype', 'embedding_name',
                                   'embedding'])):
    __slots__ = ()
    
    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,
                embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype,
                                                    embedding_name, embedding)


def build_input_features(feature_columns, mask_zero=True, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            input_features[fc.name] = Input(
                shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            input_features[fc.name] = Input(
                shape=(fc.dimension,), name=prefix + fc.name, dtype=fc.dtype)
        elif isinstance(fc, VarLenSparseFeat):
            input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + fc.name,
                                            dtype=fc.dtype)
            if not mask_zero:
                input_features[fc.name + "_seq_length"] = Input(shape=(
                    1,), name=prefix + 'seq_length_' + fc.name)
                input_features[fc.name + "_seq_max_length"] = fc.maxlen
        else:
            raise TypeError("Invalid feature column type,got", type(fc))
    
    return input_features


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True, initializer_mode="random_normal"):
    if embedding_size == 'auto':
        print("Notice:Do not use auto embedding in models other than DCN")
        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                           embeddings_initializer=get_embedding_initializer(
                                                               initializer_mode=initializer_mode,
                                                               mean=0.0,
                                                               stddev=init_std,
                                                               seed=seed),
                                                           embeddings_regularizer=l2(l2_reg),
                                                           name=prefix + '_emb_' + feat.name) for feat in
                            sparse_feature_columns}
    else:
        
        sparse_embedding = {feat.embedding_name: Embedding(feat.dimension, embedding_size,
                                                           embeddings_initializer=get_embedding_initializer(
                                                               initializer_mode=initializer_mode,
                                                               mean=0.0,
                                                               stddev=init_std,
                                                               seed=seed),
                                                           embeddings_regularizer=l2(
                                                               l2_reg),
                                                           name=prefix + '_emb_' + feat.name) for feat in
                            sparse_feature_columns}
    
    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            if embedding_size == "auto":
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, 6 * int(pow(feat.dimension, 0.25)),
                                                                  embeddings_initializer=get_embedding_initializer(
                                                                      initializer_mode=initializer_mode,
                                                                      mean=0.0,
                                                                      stddev=init_std,
                                                                      seed=seed),
                                                                  embeddings_regularizer=l2(
                                                                      l2_reg),
                                                                  name=prefix + '_seq_emb_' + feat.name,
                                                                  mask_zero=seq_mask_zero)
            
            else:
                sparse_embedding[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                                  embeddings_initializer=get_embedding_initializer(
                                                                      initializer_mode=initializer_mode,
                                                                      mean=0.0,
                                                                      stddev=init_std,
                                                                      seed=seed),
                                                                  embeddings_regularizer=l2(
                                                                      l2_reg),
                                                                  name=prefix + '_seq_emb_' + feat.name,
                                                                  mask_zero=seq_mask_zero)
    return sparse_embedding


def get_embedding_initializer(initializer_mode="random_normal", mean=0.0, stddev=0.01, minval=-0.05, maxval=0.05,
                              seed=1024):
    if initializer_mode == "random_normal":
        initializer = RandomNormal(mean=mean, stddev=stddev, seed=seed)
    elif initializer_mode == "truncated_normal":
        initializer = TruncatedNormal(mean=mean, stddev=stddev, seed=seed)
    elif initializer_mode == "glorot_normal":
        initializer = GlorotNormal(seed=seed)
    elif initializer_mode == "random_uniform":
        initializer = RandomUniform(minval=minval, maxval=maxval, seed=seed)
    else:
        raise Exception("Don't support embedding initializer_mode: ", initializer_mode)
    return initializer


def create_embedding_matrix(feature_columns, l2_reg, init_std, seed, embedding_size, prefix="", seq_mask_zero=True,
                            initializer_mode="random_normal"):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat) and x.embedding, feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size,
                                            init_std, seed, l2_reg, prefix=prefix + 'sparse',
                                            seq_mask_zero=seq_mask_zero,
                                            initializer_mode=initializer_mode)
    return sparse_emb_dict


def get_linear_logit(features, feature_columns, units=1, use_bias=True, l2_reg=0, init_std=0.0001, seed=1024,
                     prefix='linear'):
    linear_emb_list = [
        input_from_feature_columns(features, feature_columns, 1, l2_reg, init_std, seed, prefix=prefix + str(i))[0] for
        i in range(units)]
    _, dense_input_list = input_from_feature_columns(features, feature_columns, 1, l2_reg, init_std, seed,
                                                     prefix=prefix)
    
    linear_logit_list = []
    for i in range(units):
        if len(linear_emb_list[0]) > 0 and len(dense_input_list) > 0:
            sparse_input = Utils.concat_func(linear_emb_list[i])
            dense_input = Utils.concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias)([sparse_input, dense_input])
        elif len(linear_emb_list[0]) > 0:
            sparse_input = Utils.concat_func(linear_emb_list[i])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = Utils.concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias)(dense_input)
        else:
            raise NotImplementedError
        linear_logit_list.append(linear_logit)
    
    return Utils.concat_func(linear_logit_list)


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=()):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list and fc.embedding:
            if fc.use_hash:
                lookup_idx = Hash(fc.dimension, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]
            
            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))
    
    return embedding_vec_list


def get_dense_input(features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])
    return dense_input_list


def input_from_feature_columns(features, feature_columns, embedding_size, l2_reg, init_std, seed, prefix='',
                               seq_mask_zero=True, support_dense=True, initializer_mode="random_normal"):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    
    embedding_dict = create_embedding_matrix(feature_columns, l2_reg, init_std, seed, embedding_size, prefix=prefix,
                                             seq_mask_zero=seq_mask_zero, initializer_mode=initializer_mode)
    sparse_embedding_list = embedding_lookup(
        embedding_dict, features, sparse_feature_columns)
    dense_value_list = get_dense_input(features, feature_columns)
    if not support_dense and len(dense_value_list) > 0:
        raise ValueError("DenseFeat is not supported in dnn_feature_columns")
    
    return sparse_embedding_list, dense_value_list
