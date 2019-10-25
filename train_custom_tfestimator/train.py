#!/usr/bin/env python3
"""
main file to run the training process
"""
import sys
from glob import glob

import tensorflow as tf
from train_custom_tfestimator.config import INT_MHE_FEAT, RESPONSE_FEAT, INT_OHE_FEAT, INT_NUM_FEAT, NUM_USERTAG
from train_custom_tfestimator.data_generator import get_eval_inputs, get_train_inputs, get_label_from_TFrecord
from train_custom_tfestimator.model import Loss, DNN

from sklearn.metrics import roc_auc_score, average_precision_score

PREDICT = tf.estimator.ModeKeys.PREDICT
EVAL = tf.estimator.ModeKeys.EVAL
TRAIN = tf.estimator.ModeKeys.TRAIN


def model_fn(features, labels, mode, params):
    """Model function used in the estimator.

    Args:
        features (Tensor):  Input features to the model.
        labels (Tensor):    Labels tensor for training and evaluation.
        mode (ModeKeys):    Specifies if training, evaluation or prediction.
        params (dict):      Dictionary of Hyper-parameters.

    Returns:
        (EstimatorSpec):    Model to be run by Estimator.
    """

    # Define model's architecture
    if params['model'] == 'DNN':
        y_prob, logit, l1_reg, l2_reg = DNN.model(features, params, mode)
    else:
        tf.logging.fatal('Params model setting wrong with {0}'.format(params['model']))
        sys.exit(1)

    export_outputs = None
    y_prediction = tf.reshape(y_prob, [-1])
    predictions_dict = {'predicted': y_prediction}

    if mode in (TRAIN, EVAL):

        labels_flatten = tf.reshape(tf.cast(labels, dtype=tf.int32), [-1])
    
        auc = tf.compat.v1.metrics.auc(labels_flatten, y_prediction)
        apk = tf.compat.v1.metrics.average_precision_at_k(tf.cast(labels_flatten, dtype=tf.int64),
                                        y_prediction,
                                        k=params['batch_size'])
        precision = tf.compat.v1.metrics.precision(labels_flatten, y_prediction)
        recall = tf.compat.v1.metrics.recall(labels_flatten, y_prediction)

        metrics = {'eval_auc': auc, 'eval_avgpk': apk, 'eval_precision': precision, 'eval_recall': recall}
        tf.compat.v1.summary.scalar('train_auc', auc[1])
        tf.compat.v1.summary.scalar('train_avgpk', apk[1])
        tf.compat.v1.summary.scalar('train_precision', precision[1])
        tf.compat.v1.summary.scalar('train_recall', recall[1])

        loss = Loss.loss_func(labels, y_prob, logit, l1_reg, l2_reg, params)

        if mode == TRAIN:
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                global_step = tf.compat.v1.train.get_global_step()
                learning_rate = tf.compat.v1.train.exponential_decay(learning_rate=params['learning_rate'],
                                                           global_step=global_step,
                                                           decay_steps=params['decay_step'],
                                                           decay_rate=params['decay_rate'])

                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1)
                trainable_variables = tf.compat.v1.trainable_variables()
                gradients = tf.gradients(loss, trainable_variables)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                train_op = optimizer.apply_gradients(zip(clipped_gradients, trainable_variables),
                                                     global_step=global_step)
                update_global_step = tf.compat.v1.assign(global_step, global_step + 1, name='update_global_step')
                final_train_op = tf.group(train_op, update_global_step)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions_dict,
                loss=loss,
                train_op=final_train_op,
                export_outputs=export_outputs)

        else:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions_dict,
                loss=loss,
                eval_metric_ops=metrics,
                export_outputs=export_outputs)

    elif mode == PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict)

    else:
        tf.logging.fatal('Training MODE setting is wrong: {0}'.format(mode))
        sys.exit(1)


def build_estimator(config, params):
    """Build the estimator based on the given config and params.

    Args:
        config (RunConfig): RunConfig object that defines how to run the Estimator.
        params (dict):      Dictionary of Hyper-parameters.
    """
    if glob(params['save_dir'] + params['load_dir'] + '/*checkpoint*'):
        return tf.estimator.Estimator(
            model_fn=model_fn,
            config=config,
            params=params,
            model_dir=params['save_dir'] + params['store_dir'],
            warm_start_from=params['save_dir'] + params['load_dir']
            )
    else:
        return tf.estimator.Estimator(
            model_fn=model_fn,
            config=config,
            params=params,
            model_dir=params['save_dir'] + params['store_dir'],
            )

def build_spec(params, train_file, eval_file):
    """Create the training spec and testing spec of estimator

    Args:
        params (dict):     Dictionary of Hyper-parameters.
        train_file (list): Lists of training TFrecord Files
        eval_file (list):  Lists of evaluation TFrecord Files

    Returns:
        train_spec (tf.estimator.TrainSpec): training spec
        eval_spec (tf.estimator.EvalSpec):   evaluation spec
    """
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: get_train_inputs(train_file, params['batch_size'],
                                          params['num_epochs'],
                                          params['oversample'] ),
        max_steps=params['max_training_steps'])

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: get_eval_inputs(eval_file, params['batch_size']),
        steps=params['num_eval_steps'],
        start_delay_secs=params['start_delay_secs'],
        throttle_secs=params['throttle_secs'])
    return train_spec, eval_spec


def train_and_evaluate(model_estimator, train_spec, eval_spec):
    tf.logging.info('======= START Training & Evaluation =======')
    tf.estimator.train_and_evaluate(model_estimator, train_spec, eval_spec)
    tf.logging.info('======= DONE Training & Evaluation ========')


def test(model_estimator, test_file, batch_size):
    tf.logging.info('============== START Testing ==============')

    def input_function(): return get_eval_inputs(test_file, batch_size)

    evaluate_results = model_estimator.predict(input_fn=input_function)
    tf.logging.info('============== DONE Testing  ==============')
    return evaluate_results


def export_saved_model(model_estimator, params, serving_input_receiver_fn):
    tf.logging.info('======= START Saved_model exporting =======')
    model_estimator.export_savedmodel(
        export_dir_base=params['save_dir'] + params['model'],
        serving_input_receiver_fn=serving_input_receiver_fn)
    tf.logging.info('======= DONE Saved_model exporting ========')


def saved_model_input_function():
    """Customized receiver function for saved_model serving

    Returns:
        class : tf.estimator.export.ServingInputReceiver
    """
    inference_spec = {}

    for feature in INT_MHE_FEAT:
        inference_spec.update({feature: tf.placeholder(tf.int64, shape=(None, None), name=feature)})

    for feature in RESPONSE_FEAT + INT_OHE_FEAT + INT_NUM_FEAT:
        inference_spec.update({feature: tf.placeholder(tf.int64, shape=(None, 1), name=feature)})


    return tf.estimator.export.ServingInputReceiver(inference_spec, inference_spec)


def test_model(model_estimator, test_file, params):
    """test Saved_model

    Args:
        model_estimator (tf.estimator): model estimator
        test_file (list): List of files for testing
        params (dict): Hyper-parameters (argparse object).

    Returns:
        auc(float): auc
        ap(float): ap
    """
    tf.logging.info('======== START Saved_model testing ========')
    prediction = []
    y = []

    # Log the testing set AUC
    evaluate_results = test(model_estimator, test_file, params['batch_size'])

    for probability in evaluate_results:
        prediction.append(probability['predicted'])

    for file in test_file:
        y_temp = get_label_from_TFrecord(file)
        y += y_temp

    # Report Area under Curve
    try:
        tf.logging.info('length of testing set :  {}'.format(len(y)))
        auc = roc_auc_score(y, prediction)
        tf.logging.info('Testing set AUC :        {}'.format(auc))
    except Exception as err:
        auc = -1
        tf.logging.error('Testing set AUC error : {}'.format(str(err)))

    # Report Average precision
    try:
        ap = average_precision_score(y, prediction)
        tf.logging.info('Testing set AP :         {}'.format(ap))
    except Exception as err:
        ap = -1
        tf.logging.error('Testing set AP error :  {}'.format(err))

    tf.logging.info('======== DONE Saved_model testing =========')

    return auc, ap


def make_oneshot_inference(model_dir, model_name):
    """ Make one-shot prediction."""

    instance_dict = {
        'click': [[0]],
        'usertag': [[0] * NUM_USERTAG],
        'weekday': [[0]],
        'region': [[0]],
        'city': [[0]],
        'adexchange': [[0]],
        'slotformat': [[0]],
        'hour': [[0]],
        'slotwidth': [[0]],
        'slotheight': [[0]],
        'slotvisibility': [[0]],
        'slotprice': [[0]]
    }

    saved_model_dir = glob(model_dir + model_name + '/*')
    predict_fn = tf.contrib.predictor.from_saved_model(saved_model_dir[-1])
    predictions = predict_fn(instance_dict)

    return predictions
