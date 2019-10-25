#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def loss_func(y, y_prob, logit, l1_reg, l2_reg, params):
    """loss function for models

    Args:
        y      (tf.int64): true class labels
        y_prob (tf.float32): predicted probability of class 1
        logit  (tf.float32): logit of prediction
        l1_reg (tf.float32): L1 regularization
        l2_reg (tf.float32): L2 regularization
        class_weight(tf.float32): batch class weight
        params (dict): collection of arguments parameters

    Returns:
        loss (tf.float32) : batch loss
    """
    with tf.name_scope("loss"):
        mse = tf.reduce_mean(tf.square(tf.subtract(y, y_prob)))
        xloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit))
        weighted_xloss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=y,
                                                                                 logits=logit,
                                                                                 pos_weight=params['class_ratio']))
        focal_loss = tf.reduce_mean(focal_loss_fn(y, logit, params))

        # show loss in Tensorboard
        tf.compat.v1.summary.scalar('mse',        mse,           family='Loss')
        tf.compat.v1.summary.scalar('xloss',      xloss,         family='Loss')
        tf.compat.v1.summary.scalar('focal_loss', focal_loss,    family='Loss')

        if params['loss'] == 10:
            loss = mse
        elif params['loss'] == 11:
            loss = mse + params['beta'] * l1_reg
        elif params['loss'] == 12:
            loss = mse + params['beta'] * l2_reg
        elif params['loss'] == 20:
            loss = xloss
        elif params['loss'] == 21:
            loss = xloss + params['beta'] * l1_reg
        elif params['loss'] == 22:
            loss = xloss + params['beta'] * l2_reg
        elif params['loss'] == 30:
            loss = weighted_xloss
        elif params['loss'] == 31:
            loss = weighted_xloss + params['beta'] * l1_reg
        elif params['loss'] == 32:
            loss = weighted_xloss + params['beta'] * l2_reg
        elif params['loss'] == 40:
            loss = focal_loss
        elif params['loss'] == 41:
            loss = focal_loss + params['beta'] * l1_reg
        elif params['loss'] == 42:
            loss = focal_loss + params['beta'] * l2_reg
        else:
            loss = mse
        return loss

def focal_loss_fn(labels, logits, params):
    """Focal loss for imbalanced data

    Args:
        y (Tensor): Ground truth
        y_pred (Tensor): Predicted probability
        params (dict): Dict to store hyperparameter

    Returns:
        focal_loss (Tensor): Single value tensor of focal loss

    
    p = y * y_pred + tf.subtract(1., y) * tf.subtract(1., y_pred)
    fl = -params['alpha'] * (1. - p) ** params['gamma'] * tf.log(p)
    focal_loss = tf.reduce_sum(fl, axis=0)
    """
    try:
      labels.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError("logits and labels must have the same shape (%s vs %s)" %
                       (logits.get_shape(), labels.get_shape()))

    log_weight = math_ops.divide( (params['alpha'] * labels * math_ops.exp(-logits * params['gamma'])) - labels + 1,
                                  (1 + math_ops.exp(-logits))**params['gamma'] )
    
    focal_loss = math_ops.add(math_ops.divide(1-labels, (1 + math_ops.exp(-logits))**params['gamma']) * logits,
                            log_weight * (math_ops.log1p(math_ops.exp(-math_ops.abs(logits))) +
                            nn_ops.relu(-logits)))

    return focal_loss
