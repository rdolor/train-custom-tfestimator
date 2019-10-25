#!/usr/bin/env python3

import sys

import tensorflow as tf

sys.path.append('..')
from train_custom_tfestimator.config import INT_OHE_FEAT, ONEHOT_FEAT_NUM_LIST, INT_MHE_FEAT, MULTIHOT_FEAT_NUM_LIST, INT_NUM_FEAT

def model(inputs, params, mode):
    """Return the output operation following the DNN architecture.

    Args:
        inputs (Tensor): input tensor
        params (dict): parameter to be passed to model
        mode (str):  mode of TRAIN, EVAL or TEST

    Returns:
        {key(feature_name): value([ # items,
                                    # embedding units,
                                    placeholder,
                                    weight,
                                    embedding lookup weight])}
    """

    # create dictionary for storing non-embedding features
    for feature_name in INT_NUM_FEAT:
        inputs[feature_name] = tf.cast(tf.reshape(inputs[feature_name], [-1, 1]), dtype=tf.float32, name=feature_name)

    # create dictionary for storing one-hot embedding features
    one_hot_dict = {}
    for feature_name, num_item, num_embed_units in zip(INT_OHE_FEAT, ONEHOT_FEAT_NUM_LIST, params['one_hot_units']):
        if feature_name not in one_hot_dict.keys():
            one_hot_dict[feature_name] = []
        one_hot_dict[feature_name].extend([num_item, num_embed_units])

    for f_name in one_hot_dict.keys():
        inputs[f_name] = tf.cast(tf.reshape(inputs[f_name], [-1, 1]), dtype=tf.int32, name=f_name)
        one_hot_dict[f_name].append(tf.compat.v1.get_variable(f_name + "_emb_w",
                                                    [one_hot_dict[f_name][0], one_hot_dict[f_name][1]],
                                                    initializer=tf.random_normal_initializer(), dtype=tf.float32))
        one_hot_dict[f_name].append(tf.nn.embedding_lookup(one_hot_dict[f_name][2], inputs[f_name]))

    # create dictionary for storing multi-hot embedding features
    multi_hot_dict = {}
    for feature_name, num_item, num_embed_units in zip(INT_MHE_FEAT, MULTIHOT_FEAT_NUM_LIST, params['multi_hot_units']):
        if feature_name not in one_hot_dict.keys():
            multi_hot_dict[feature_name] = []
        multi_hot_dict[feature_name].extend([num_item, num_embed_units])

    for f_name in multi_hot_dict.keys():
        multi_hot_dict[f_name].append(tf.compat.v1.get_variable(f_name + "_emb_w",
                                                      [multi_hot_dict[f_name][0], multi_hot_dict[f_name][1]],
                                                      initializer=tf.random_normal_initializer(), dtype=tf.float32))
        multi_hot_dict[f_name].append(tf.reduce_sum(tf.nn.embedding_lookup(multi_hot_dict[f_name][2], inputs[f_name]), 1))

    # Set numerical list features
    numerical_feature_list = INT_NUM_FEAT 
    x = inputs[numerical_feature_list[0]]
    for f_name in numerical_feature_list[1::]:
        x = tf.concat([x, tf.cast(tf.reshape(inputs[f_name],[tf.shape(inputs[f_name])[0],-1]), dtype=tf.float32)], axis=1)
    for f_name in one_hot_dict.keys():
        x = tf.concat([x, tf.cast(tf.reshape(one_hot_dict[f_name][3], [-1, one_hot_dict[f_name][1]]), dtype=tf.float32)], axis=1, name='x')
    for f_name in multi_hot_dict.keys():
        x = tf.concat([x, tf.reshape(multi_hot_dict[f_name][3], [-1, multi_hot_dict[f_name][1]] )], axis=1, name='x')

    x = tf.reshape(x, [-1,params['num_features']])
    training = bool(mode == tf.estimator.ModeKeys.TRAIN)
    bn = tf.layers.batch_normalization(inputs=x,axis=1, training=training, name='batch_norm')
    dnn = tf.layers.dense(bn, params['hidden_units'][0], activation=tf.nn.elu, name='dnn1')
    for hidden_unit_idx in range(1, len(params['hidden_units'])):
        dnn = tf.layers.dense(dnn, params['hidden_units'][hidden_unit_idx], activation=tf.nn.elu,name='dnn' + str(hidden_unit_idx + 1))
        dnn = tf.layers.dropout(dnn, rate=params['dropout_rate'], training=training, name='dropout' + str(hidden_unit_idx + 1))

    logit = tf.reshape(tf.layers.dense(dnn, 1, activation=None, name='output_layer'), [-1, 1])
    y_prob = tf.nn.sigmoid(logit, name='prediction_node')

    l1_reg = 0
    l2_reg = 0
    for f_name in one_hot_dict.keys():
        l1_reg += tf.reduce_sum(tf.abs(one_hot_dict[f_name][3]))
        l2_reg += tf.nn.l2_loss(one_hot_dict[f_name][3])
    for f_name in multi_hot_dict.keys():
        l1_reg += tf.reduce_sum(tf.abs(multi_hot_dict[f_name][3]))
        l2_reg += tf.nn.l2_loss(multi_hot_dict[f_name][3])

    return y_prob, logit, l1_reg, l2_reg
