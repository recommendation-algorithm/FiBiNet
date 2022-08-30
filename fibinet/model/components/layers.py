#!/usr/bin/env python
# -*- coding:utf-8 -*-

import itertools

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.initializers import Zeros, glorot_normal
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2

from fibinet.common.utils import Utils


class Hash(Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """
    
    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(Hash, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Hash, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            x = tf.as_string(x, )
        try:
            hash_x = tf.string_to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                   name=None)  # weak hash
        except:
            hash_x = tf.strings.to_hash_bucket_fast(x, self.num_buckets if not self.mask_zero else self.num_buckets - 1,
                                                    name=None)  # weak hash
        if self.mask_zero:
            mask_1 = tf.cast(tf.not_equal(x, "0"), 'int64')
            mask_2 = tf.cast(tf.not_equal(x, "0.0"), 'int64')
            mask = mask_1 * mask_2
            hash_x = (hash_x + 1) * mask
        return hash_x
    
    def compute_mask(self, inputs, mask=None):
        return None
    
    def get_config(self, ):
        config = super(Hash, self).get_config()
        config.update({
            "num_buckets": self.num_buckets,
            "mask_zero": self.mask_zero
        })
        return config


class Linear(Layer):
    
    def __init__(self, l2_reg=0.0, mode=0, use_bias=True, **kwargs):
        self.l2_reg = l2_reg
        self.mode = mode
        self.use_bias = use_bias
        super(Linear, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if self.use_bias:
            self.bias = self.add_weight(name='linear_bias',
                                        shape=(1,),
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=True)
        self.dense = tf.keras.layers.Dense(units=1, activation=None, use_bias=False,
                                           kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        super(Linear, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, inputs, **kwargs):
        if self.mode == 0:
            sparse_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keepdims=True)
        elif self.mode == 1:
            dense_input = inputs
            linear_logit = self.dense(dense_input)
        else:
            sparse_input, dense_input = inputs
            linear_logit = tf.reduce_sum(sparse_input, axis=-1, keepdims=False) + self.dense(dense_input)
        if self.use_bias:
            linear_logit = linear_logit + self.bias
        return linear_logit
    
    def compute_output_shape(self, input_shape):
        return None, 1
    
    def get_config(self, ):
        config = super(Linear, self).get_config()
        config.update({
            "l2_reg": self.l2_reg,
            'mode': self.mode,
            "use_bias": self.use_bias,
        })
        return config


class SimpleLayerNormalization(Layer):
    """Applies layer normalization."""
    
    def __init__(self, is_share_params, **kwargs):
        super(SimpleLayerNormalization, self).__init__(**kwargs)
        self.is_share_params = is_share_params
        self.field_size = -1
        self.embedding_size = -1
        return
    
    def build(self, input_shape):
        self.field_size = input_shape[-2].value
        self.embedding_size = input_shape[-1].value
        if self.is_share_params:
            self.scale = self.add_weight("layer_norm_scale", [self.embedding_size],
                                         initializer=tf.ones_initializer())
            self.bias = self.add_weight("layer_norm_bias", [self.embedding_size],
                                        initializer=tf.zeros_initializer())
        else:
            self.scale = self.add_weight("layer_norm_scale", [1, self.field_size, self.embedding_size],
                                         initializer=tf.ones_initializer())
            self.bias = self.add_weight("layer_norm_bias", [1, self.field_size, self.embedding_size],
                                        initializer=tf.zeros_initializer())
        super(SimpleLayerNormalization, self).build(input_shape)
        return
    
    def call(self, x, epsilon=1e-6):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        outputs = norm_x * self.scale + self.bias
        return outputs
    
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape
    
    def get_config(self, ):
        config = super(SimpleLayerNormalization, self).get_config()
        config.update({
            "is_share_params": self.is_share_params,
        })
        return config


class DenseEmbeddingLayer(Layer):
    def __init__(self, embedding_size, init_std, embedding_l2_reg=0.0, seed=2018):
        super(DenseEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.init_std = init_std
        self.embedding_l2_reg = embedding_l2_reg
        self.seed = seed
        self.field_size = 0
        self.embedding_weights = None
        return
    
    def build(self, input_shape):
        self.field_size = input_shape[-2].value
        self.embedding_weights = self.add_weight(name="dense_embedding_weights",
                                                 shape=(1, self.field_size, self.embedding_size),
                                                 initializer=RandomNormal(mean=0.0, stddev=self.init_std,
                                                                          seed=self.seed),
                                                 regularizer=l2(self.embedding_l2_reg))
        super(DenseEmbeddingLayer, self).build(input_shape)
        return
    
    def call(self, inputs, **kwargs):
        embedding_weights = self.embedding_weights
        embeddings = inputs * embedding_weights
        return embeddings
    
    def compute_output_shape(self, input_shape):
        output_shape = (None, self.filed_size, self.embedding_size)
        return output_shape
    
    def get_config(self, ):
        config = super(DenseEmbeddingLayer, self).get_config()
        config.update({
            "embedding_size": self.embedding_size,
            "init_std": self.init_std,
            "embedding_l2_reg": self.embedding_l2_reg,
        })
        return config


class SENETLayer(Layer):
    """SENETLayer used in FiBiNET.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Output shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.
      Arguments
        - **reduction_ratio** : Positive integer, dimensionality of the
         attention network output space.
        - **seed** : A Python integer to use as random seed.
    """
    
    def __init__(self, senet_squeeze_mode="mean", senet_squeeze_group_num=1, senet_squeeze_topk=1,
                 senet_reduction_ratio=3.0, senet_excitation_mode="vector", senet_activation="relu",
                 senet_use_skip_connection=False, senet_reweight_norm_type="none", seed=1024, **kwargs):
        self.senet_squeeze_mode = senet_squeeze_mode
        self.senet_squeeze_group_num = senet_squeeze_group_num
        self.senet_squeeze_topk = senet_squeeze_topk
        self.senet_reduction_ratio = senet_reduction_ratio
        self.senet_excitation_mode = senet_excitation_mode
        self.senet_activation = senet_activation
        self.senet_use_skip_connection = senet_use_skip_connection
        self.senet_reweight_norm_type = senet_reweight_norm_type
        self.seed = seed
        
        self.field_size = None
        self.embedding_size = None
        self.tensor_dot = None
        self.W_1 = None
        self.W_2 = None
        self.reweight_norm_layer = None
        super(SENETLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if len(input_shape) < 3:  # [batch, field, hidden_size]
            raise ValueError('A `SENETLayer` layer should be called on at least 3 inputs')
        
        self.field_size = input_shape[-2].value
        self.embedding_size = input_shape[-1].value
        self.tensor_dot = tf.keras.layers.Lambda(lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))
        
        # Squeeze & Excitation
        input_dim = self._squeeze_dimension()
        reduction_size = int(max(1.0, input_dim / self.senet_reduction_ratio))
        output_dim = self._excitation_dimension()
        self.W_1 = self.add_weight(shape=(
            input_dim, reduction_size), initializer=glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, output_dim), initializer=glorot_normal(seed=self.seed), name="W_2")
        
        if "ln" in self.senet_reweight_norm_type:
            self.reweight_norm_layer = SimpleLayerNormalization(True)
        
        super(SENETLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, inputs, training=None, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        
        # Squeeze
        Z = self._squeeze_inputs(inputs)
        
        # Excitation
        A_1 = tf.nn.relu(self.tensor_dot([Z, self.W_1]))
        A_2 = self.tensor_dot([A_1, self.W_2])
        if self.senet_activation == "relu":
            A_2 = tf.nn.relu(A_2)
        
        # Re-weight
        if "bit" in self.senet_excitation_mode:
            weights = tf.reshape(A_2, [-1, self.field_size, self.embedding_size])  # [batch, fields, embeddings]
        else:
            weights = tf.expand_dims(A_2, axis=2)  # [batch, self.fields*groups, 1]
        reweight_inputs = inputs  # [batch, fields, embeddings]
        if "group" in self.senet_excitation_mode:
            # [batch, self.fields*groups, embeddings//groups]
            reweight_inputs = tf.reshape(reweight_inputs, (-1, self.field_size * self.senet_squeeze_group_num,
                                                           self.embedding_size // self.senet_squeeze_group_num))
        outputs = tf.multiply(reweight_inputs, weights)
        if "group" in self.senet_excitation_mode:
            outputs = tf.reshape(outputs, (-1, self.field_size, self.embedding_size))
        
        # Add skip-connection
        if self.senet_use_skip_connection:
            outputs += inputs
        # Norm after re-weight
        if "ln" in self.senet_reweight_norm_type:
            outputs = self.reweight_norm_layer(outputs)
        return outputs
    
    def _squeeze_dimension(self):
        dimension = 0
        if "mean" in self.senet_squeeze_mode:
            dimension += self.field_size
        if "min" in self.senet_squeeze_mode:
            dimension += self.field_size
        if "max" in self.senet_squeeze_mode:
            dimension += self.field_size
        if "topk" in self.senet_squeeze_mode:
            dimension += self.field_size * self.senet_squeeze_topk
        if "group" in self.senet_squeeze_mode:
            dimension *= self.senet_squeeze_group_num
        if "bit" in self.senet_squeeze_mode:
            # conflict with other modes,  = concat[fields]
            dimension = self.field_size * self.embedding_size
        return dimension
    
    def _squeeze_inputs(self, inputs):
        squeeze_results = []
        if "group" in self.senet_squeeze_mode:
            inputs = tf.reshape(inputs, (-1, self.field_size * self.senet_squeeze_group_num,
                                         self.embedding_size // self.senet_squeeze_group_num))
        if "mean" in self.senet_squeeze_mode:
            squeeze_tensor = tf.reduce_mean(inputs, axis=-1)  # [batch, fields]
            squeeze_results.append(squeeze_tensor)
        if "min" in self.senet_squeeze_mode:
            squeeze_tensor = tf.reduce_min(inputs, axis=-1)  # [batch, fields]
            squeeze_results.append(squeeze_tensor)
        if "max" in self.senet_squeeze_mode:
            squeeze_tensor = tf.reduce_max(inputs, axis=-1)  # [batch, fields]
            squeeze_results.append(squeeze_tensor)
        if "topk" in self.senet_squeeze_mode:
            output_topk = tf.nn.top_k(inputs, self.senet_squeeze_topk, sorted=True)
            # [batch, fields*topk*group]
            squeeze_tensor = tf.reshape(output_topk.values, [-1, self.field_size * self.senet_squeeze_topk
                                                             * self.senet_squeeze_group_num])
            squeeze_results.append(squeeze_tensor)
        if "bit" in self.senet_squeeze_mode:
            bit_tensor = tf.reshape(inputs, [-1, self.field_size * self.embedding_size])  # [batch, fields*embeddings]
            squeeze_results.append(bit_tensor)
        
        outputs = Utils.concat_func(squeeze_results, axis=-1)  # [batch, fields * k]
        return outputs
    
    def _excitation_dimension(self):
        # default ExcitationMode: vector
        output_dim = self.field_size
        if "group" in self.senet_excitation_mode:  # ExcitationMode: group
            output_dim = self.field_size * self.senet_squeeze_group_num
        if "bit" in self.senet_excitation_mode:  # ExcitationMode = bit
            output_dim = self.field_size * self.embedding_size
        return output_dim
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def compute_mask(self, inputs, mask=None):
        return [None] * self.field_size
    
    def get_config(self, ):
        config = super(SENETLayer, self).get_config()
        config.update({
            "senet_squeeze_mode": self.senet_squeeze_mode,
            "senet_squeeze_group_num": self.senet_squeeze_group_num,
            "senet_squeeze_topk": self.senet_squeeze_topk,
            "senet_reduction_ratio": self.senet_reduction_ratio,
            "senet_excitation_mode": self.senet_excitation_mode,
            "senet_activation": self.senet_activation,
            "senet_use_skip_connection": self.senet_use_skip_connection,
            "senet_reweight_norm_type": self.senet_reweight_norm_type,
            'seed': self.seed,
            'params': self.params,
        })
        return config


class BilinearInteraction(Layer):
    """BilinearInteraction Layer used in FiBiNET.

      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Arguments
        - **str** : String, types of bilinear functions used in this layer.

        - **seed** : A Python integer to use as random seed.

      References
        - [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction]
        (https://arxiv.org/pdf/1905.09433.pdf)

    """
    
    def __init__(self, bilinear_type="interaction", dnn_units=(), dnn_activation="linear", seed=1024, **kwargs):
        self.bilinear_type = bilinear_type
        self.dnn_units = dnn_units
        self.dnn_activation = dnn_activation
        self.seed = seed
        self.field_size = -1
        self.embedding_size = -1
        self.upper_triangular_indices = []
        self.W = None
        self.W_list = None
        self.dnn_layer = None
        super(BilinearInteraction, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('A `BilinearInteraction` layer should be called '
                             'on a tensor of shape (?, fields, embeddings)')
        self.field_size = input_shape[-2].value
        self.embedding_size = input_shape[-1].value
        self.upper_triangular_indices = Utils.get_upper_triangular_indices(self.field_size)
        
        if "all" in self.bilinear_type:
            self.W = self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each":
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i)) for i in range(self.field_size - 1)]
        elif self.bilinear_type == "interaction":
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size), initializer=glorot_normal(
                seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(self.field_size), 2)]
        elif self.bilinear_type == "none":
            pass
        else:
            raise NotImplementedError
        
        if len(self.dnn_units) > 0:
            self.dnn_layer = DNNLayer(self.dnn_units, activation=self.dnn_activation, l2_reg=0.0, dropout_rate=0.0,
                                      use_bn=False)
        
        super(BilinearInteraction, self).build(input_shape)  # Be sure to call this somewhere!
        return
    
    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:  # [batch, fields, embeddings]
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        
        if self.bilinear_type == "all":
            split_inputs = tf.split(inputs, self.field_size, axis=-2)
            p = [tf.multiply(tf.tensordot(v_i, self.W, axes=(-1, 0)), v_j)
                 for v_i, v_j in itertools.combinations(split_inputs, 2)]
        elif self.bilinear_type == "each":
            split_inputs = tf.split(inputs, self.field_size, axis=-2)
            p = [tf.multiply(tf.tensordot(split_inputs[i], self.W_list[i], axes=(-1, 0)), split_inputs[j])
                 for i, j in itertools.combinations(range(self.field_size), 2)]
        elif self.bilinear_type == "interaction":
            split_inputs = tf.split(inputs, self.field_size, axis=-2)
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(split_inputs, 2), self.W_list)]
        elif self.bilinear_type == "all_high":
            results = self._full_interaction(inputs, self.W)  # [batch, 1, new_fields*embedding]
            p = [results]
        elif self.bilinear_type == "all_ip":
            results = self._full_interaction_ip(inputs, self.W)  # [batch, 1, new_fields]
            p = [results]
        elif self.bilinear_type == "none":
            p = [tf.reshape(inputs, [-1, 1, self.field_size * self.embedding_size])]
        else:
            raise NotImplementedError
        outputs = Utils.concat_func(p)  # [batch, 1, size]
        
        # Output Transformation DNN
        if len(self.dnn_units) > 0:
            outputs = self.dnn_layer(outputs)
        
        return outputs
    
    def _full_interaction(self, inputs, weights):
        """
        Cross in pairs with element-wise product and take the upper triangular part
        :param inputs: [batch, fields, embeddings]
        :return:
        """
        bilinear_inputs = tf.tensordot(inputs, weights, axes=(-1, 0))
        matrix_1 = tf.expand_dims(bilinear_inputs, axis=2)  # [batch, fields, 1, hidden_size]
        matrix_1 = tf.tile(matrix_1, [1, 1, self.field_size, 1])  # [batch, fields, fields, hidden_size]
        matrix_2 = tf.expand_dims(inputs, axis=1)  # [batch, 1, fields, hidden_size]
        matrix_2 = tf.tile(matrix_2, [1, self.field_size, 1, 1])  # [batch, fields, fields, hidden_size]
        matrix_interaction = matrix_1 * matrix_2  # [batch, fields, fields, hidden_size]
        
        matrix_interaction = tf.transpose(matrix_interaction, [1, 2, 0, 3])  # [fields, fields, batch, hidden_size]
        upper_triangular = tf.gather_nd(matrix_interaction,
                                        self.upper_triangular_indices)  # [fields*(fields-1)/2, batch, hidden_size]
        outputs = tf.transpose(upper_triangular, [1, 0, 2])  # [batch, new_fields, hidden_size]
        
        outputs = tf.reshape(outputs, [-1, 1, (self.field_size * (self.field_size - 1) // 2) * self.embedding_size])
        return outputs
    
    def _full_interaction_ip(self, inputs, weights):
        """
        Cross in pairs with inner product and take the upper triangular part
        :param inputs: [batch, fields, embeddings]
        :return:
        """
        bilinear_inputs = tf.tensordot(inputs, weights, axes=(-1, 0))  # [batch fields, hidden_size]
        matrix_interaction = tf.matmul(bilinear_inputs, inputs, transpose_b=True)  # [batch, fields, fields]
        
        matrix_interaction = tf.transpose(matrix_interaction, [1, 2, 0])  # [fields, fields, batch]
        upper_triangular = tf.gather_nd(matrix_interaction,
                                        self.upper_triangular_indices)  # [fields*(fields-1)/2, batch]
        outputs = tf.transpose(upper_triangular, [1, 0])  # [batch, new_fields]
        
        outputs = tf.reshape(outputs, [-1, 1, (self.field_size * (self.field_size - 1)) // 2])
        return outputs
    
    def compute_output_shape(self, input_shape):
        filed_size = len(input_shape)
        embedding_size = input_shape[0][-1]
        
        output_shape = (None, 1, filed_size * (filed_size - 1) // 2 * embedding_size)
        if self.bilinear_type == "all_ip":
            output_shape = (None, 1, filed_size * (filed_size - 1) // 2)
        elif self.bilinear_type == "none":
            output_shape = (None, 1, filed_size * embedding_size)
        return output_shape
    
    def get_config(self, ):
        config = super(BilinearInteraction, self).get_config()
        config.update({
            "bilinear_type": self.bilinear_type,
            "seed": self.seed,
        })
        return config


class DNNLayer(Layer):
    """The Multi Layer Perceptron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with
        shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape
        ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularized strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **seed**: A Python integer to use as random seed.
    """
    
    def __init__(self, hidden_units, activation='relu', l2_reg=0.0, dropout_rate=0.0, use_bn=False, seed=1024,
                 **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.seed = seed
        super(DNNLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        input_size = input_shape[-1].value
        hidden_units = [int(input_size)] + list(self.hidden_units)
        kernel_initializer = glorot_normal(seed=self.seed)
        bias_initializer = Zeros()
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(hidden_units[i], hidden_units[i + 1]),
                                        initializer=kernel_initializer,
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=bias_initializer,
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]
        
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]
        
        if self.activation:
            self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]
        super(DNNLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs
        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            
            if self.activation:
                fc = self.activation_layers[i](fc)
            
            fc = self.dropout_layers[i](fc, training=training)
            
            deep_input = fc
        return deep_input
    
    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape
        return tuple(shape)
    
    def get_config(self, ):
        config = super(DNNLayer, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'use_bn': self.use_bn,
            'seed': self.seed,
        })
        return config


class PredictionLayer(Layer):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary log loss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """
    
    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")
        super(PredictionLayer, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == "binary":
            x = tf.sigmoid(x)
        output = tf.reshape(x, (-1, 1))
        return output
    
    def compute_output_shape(self, input_shape):
        return None, 1
    
    def get_config(self, ):
        config = super(PredictionLayer, self).get_config()
        config.update({
            'task': self.task,
            'use_bias': self.use_bias,
        })
        return config


def activation_layer(activation):
    if activation == "prelu":
        activation = tf.keras.layers.PReLU
    elif activation == "leaky_relu":
        activation = tf.keras.layers.LeakyReLU
    
    if isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, Layer):
        act_layer = activation()
    else:
        raise ValueError("Invalid activation,found %s.You should use a str or a Activation Layer Class" % activation)
    return act_layer
