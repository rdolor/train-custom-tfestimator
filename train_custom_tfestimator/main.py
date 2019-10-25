#!/usr/bin/env python3
"""
Main file to run the training process.
"""
import os

import nni
import tensorflow as tf

from glob import glob
from train_custom_tfestimator.argparser import parse_args
from train_custom_tfestimator.config import TRAIN_DATASET_PATTERN, EVAL_DATASET_PATTERN, TEST_DATASET_PATTERN, NUM_OF_CORES
from train_custom_tfestimator.train import build_estimator, build_spec, train_and_evaluate, export_saved_model, saved_model_input_function, \
    test_model
from train_custom_tfestimator.utils import write_new_training_to_csv, update_training_status_to_csv, write_error, find_training_data, \
    find_data_size


def main():

    # Read parameters and input data
    params = parse_args()

    # Locate the training/evaluation/testing data files
    train_file = find_training_data(params['start_date'], params['train_period'], params['train_data_path'], TRAIN_DATASET_PATTERN)
    eval_file  = find_training_data(params['start_date'], params['train_period'], params['train_data_path'], EVAL_DATASET_PATTERN)
    test_file  = find_training_data(params['start_date'], params['train_period'], params['train_data_path'], TEST_DATASET_PATTERN)
    if train_file == [-1] or eval_file == [-1] or test_file == [-1]:
        os._exit(1)

    tf.logging.info("Training file   : {}".format(train_file))
    tf.logging.info("Evaluation file : {}".format(eval_file))
    tf.logging.info("Testing file    : {}".format(test_file))

    num_records_train = find_data_size(train_file)
    num_records_eval  = find_data_size(eval_file)

    # early break
    if num_records_train == 0:
        tf.logging.info("Can not find any training data on {0}".format(params['start_date']))
        return

    # Create model directory
    if not os.path.exists(params['save_dir'] + params['store_dir']):
        os.makedirs(params['save_dir'] + params['store_dir'])
        
    params['num_training_steps'] = int(num_records_train/params['batch_size'])
    params['max_training_steps'] = int(num_records_train/params['batch_size'] * params['num_epochs'])
    params['num_eval_steps']     = int(num_records_eval/params['batch_size'])

    tf.logging.info('==================== Model params ======================')
    tf.logging.info('model name:            {0}'.format(params['model_name']))
    tf.logging.info('model:                 {0}'.format(params['model']))
    tf.logging.info('loss:                  {0}'.format(params['loss']))
    tf.logging.info('alpha:                 {0}'.format(params['alpha']))
    tf.logging.info('beta:                  {0}'.format(params['beta']))
    tf.logging.info('gamma:                 {0}'.format(params['gamma']))
    tf.logging.info('num_features:          {0}'.format(params['num_features']))
    tf.logging.info('==================== Training params ===================')
    tf.logging.info('has_gpu:               {0}'.format(params['has_gpu']))
    tf.logging.info('is_test:               {0}'.format(params['is_test']))
    tf.logging.info('num_records_train:     {0}'.format(num_records_train))
    tf.logging.info('num_records_eval:      {0}'.format(num_records_eval))
    tf.logging.info('train_batch_size:      {0}'.format(params['batch_size']))
    tf.logging.info('number of epochs:      {0}'.format(params['num_epochs']))
    tf.logging.info('num_training_steps:    {0}'.format(params['num_training_steps']))
    tf.logging.info('max_training_steps:    {0}'.format(params['max_training_steps']))
    tf.logging.info('num_eval_steps:        {0}'.format(params['num_eval_steps']))
    tf.logging.info('=================== Directory params ===================')
    tf.logging.info('saving directory:      {0}'.format(params['save_dir']))
    tf.logging.info('loading directory:     {0}'.format(params['load_dir']))
    tf.logging.info('storing directory:     {0}'.format(params['store_dir']))
    tf.logging.info('=========================================================')

    # Initialize summary to csv
    write_new_training_to_csv(params)

    # Setup estimator configs
    model_dir = params['save_dir'] + params['store_dir']

    sess_config = tf.ConfigProto(intra_op_parallelism_threads=NUM_OF_CORES,
                                 inter_op_parallelism_threads=2,
                                 allow_soft_placement=True,
                                 device_count={'CPU': NUM_OF_CORES})
    if params['has_gpu'] == 1:
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9

    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    save_summary_steps=params['save_summary_steps'],
                                    log_step_count_steps=params['log_step_count_steps'],
                                    save_checkpoints_steps=params['num_training_steps'],
                                    session_config=sess_config)

    # Setup the Estimator
    model_estimator = build_estimator(config, params)

    # Setup and start training and validation
    train_spec, eval_spec = build_spec(params, train_file, eval_file)
    train_and_evaluate(model_estimator, train_spec, eval_spec)
    export_saved_model(model_estimator, params, saved_model_input_function)

    # Test testing set with Saved_model
    if params['is_test'] and config.is_chief:
        saved_model_timestamp_dir = sorted(os.listdir(params['save_dir'] + params['model']))[-1]

        auc, ap = test_model(model_estimator, test_file, params)

        # Update result to csv
        try:
            update_training_status_to_csv(params, round(auc, 4), round(ap, 4), saved_model_timestamp_dir)
        except:
            write_error(params)

        # Report NNI tuning
        if params['tuning'] == 1:
            nni.report_final_result(ap)
    tf.logging.info("DONE ALL.")


if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
