#!/usr/bin/env python3
"""
Function for generator of data.Dataset API
"""

import tensorflow as tf

from src.config import NUM_OF_CORES, FEATURE_SPEC, SHUFFLE_BUFFER_SIZE, FEATURE_LIST


def extract_tfrecords(data_record):
    """
    Extracts tf records from dictionary to tuples of training & testing dataset

    Args:
        data_record (TFrecords): TFrecords of feaature & labels

    Returns:
        tuple: (sample, label)  sample: dictionary of x, y: int of binary y
    """

    data = tf.io.parse_single_example(data_record, FEATURE_SPEC)
    for keys in data.keys():
        data[keys] = tf.reshape(tf.sparse.to_dense(data[keys]), [-1])
    return data


# sampling parameters use it wisely
oversampling_coef =  0.9 # if equal to 0 then oversample_classes() always returns 1
undersampling_coef = 0.9 # if equal to 0 then undersampling_filter() always returns True

def oversample_classes(example):
    """
    Returns the number of copies of given example
    """
    class_prob = tf.cond(tf.equal(tf.reshape(example['click'],[]), 1), lambda:0.01, lambda:0.99)
    class_target_prob = 0.5

    prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)

    prob_ratio = prob_ratio ** oversampling_coef
    prob_ratio = tf.maximum(prob_ratio, 1)
    repeat_count = tf.floor(prob_ratio)
    repeat_residual = prob_ratio - repeat_count # a number between 0-1
    residual_acceptance = tf.less_equal(
                        tf.random_uniform([], dtype=tf.float32), repeat_residual
    )

    residual_acceptance = tf.cast(residual_acceptance, tf.int64)
    repeat_count = tf.cast(repeat_count, dtype=tf.int64)
    return repeat_count + residual_acceptance


def undersampling_filter(example):
    """
    Computes if given example is rejected or not.
    """
    class_prob = tf.cond(tf.equal(tf.reshape(example['click'],[]), 1), lambda:0.01, lambda:0.99)
    class_target_prob = 0.5
    prob_ratio = tf.cast(class_target_prob/class_prob, dtype=tf.float32)
    prob_ratio = prob_ratio ** undersampling_coef
    prob_ratio = tf.minimum(prob_ratio, 1.0)

    acceptance = tf.less_equal(tf.random_uniform([], dtype=tf.float32), prob_ratio)
    # predicate must return a scalar boolean tensor
    return acceptance


def get_train_inputs(train_file, batch_size, num_epochs, oversample = True):
    """Return the input function to get the training data.

    Args:
        batch_size   (int): Batch size of training iterator that is returned
                            by the input function.
        train_file (array): Training data as (inputs, labels).
        num_epochs   (int): Number of epochs

    Returns:
        DataSet: A tensorflow DataSet object to represent the training input
                 pipeline.
    """
    with tf.name_scope("tfrecord_train_reader"):
        dataset = tf.data.TFRecordDataset(train_file)
        dataset = dataset.map(extract_tfrecords, num_parallel_calls=NUM_OF_CORES)

        if oversample:
            dataset = dataset.flat_map(
                lambda x: tf.data.Dataset.from_tensors(x).repeat(oversample_classes(x)))

            dataset = dataset.filter(undersampling_filter)

        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=False).repeat(
            tf.cast(num_epochs, dtype=tf.int64))
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=dict(zip(FEATURE_LIST, [[None]] * len(FEATURE_LIST))))


        dataset = dataset.map(parse_to_label_tuples, num_parallel_calls=NUM_OF_CORES)
        dataset = dataset.prefetch(1)
        #iterator = dataset.make_one_shot_iterator()
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_dataset = iterator.get_next()
        return next_dataset


def get_eval_inputs(eval_file, batch_size, num_epochs=1):
    """Return the input function to get the validation data.

    Args:
        batch_size (int): Batch size of validation iterator that is returned
                          by the input function.
        eval_file ((array, array): CTR test data as (inputs, labels).
        num_epochs (int): Number of epoch
    Returns:
        DataSet: Tensorflow DataSet object to represent the validation input pipeline.
    """
    with tf.name_scope("tfrecord_eval_reader"):
        dataset = tf.data.TFRecordDataset(eval_file)
        dataset = dataset.map(extract_tfrecords, num_parallel_calls=NUM_OF_CORES)
        dataset = dataset.repeat(tf.cast(num_epochs, dtype=tf.int64))
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=dict(zip(FEATURE_LIST, [[None]] * len(FEATURE_LIST))))
        dataset = dataset.map(parse_to_label_tuples, num_parallel_calls=NUM_OF_CORES)
        dataset = dataset.prefetch(1)
        #iterator = dataset.make_one_shot_iterator()
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_dataset = iterator.get_next()
        return next_dataset

def parse_to_label_tuples(data_record):
    """Transform dictionary to (x,y) tuple

    Args:
        data_record (tf.dataset): tf.dataset which contains dictionary of feature tensor

    Returns:
        tuple: (sample, label)  sample: dictionary of x, y: int of binary y
    """

    label = tf.reshape(tf.cast(data_record['click'], tf.float32), [-1, 1])
    del data_record["click"]
    return data_record, label

def get_label_from_TFrecord(file):
    y = []
    record_iterator = tf.python_io.tf_record_iterator(path=file)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        y.append(example.features.feature['click'].int64_list.value[0])
    return y