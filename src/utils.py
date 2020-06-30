#!/usr/bin/env python3
"""
Contains methods for
    - preprocessing the datasets (data_generator.py) and
    - calculating the performance metrics used to evaluate the model(s).
"""
import csv
import os
import json
import re
from datetime import datetime, timedelta
from glob import glob

import tensorflow as tf

from src.config import FILE_DATE_FORMAT, FILE_DATE_REGULAR_EXP, OUTPUT_CSV_FILE, CSV_FIELD_NAME


def parse_date(start_d):
    """parse date format from Y.m.D.H.M into datetime format

    Args:
        start_d (str): Either format 'Y.m.D.H.M' or 'now'

    Returns:
        class: a datetime object
    """
    if not isinstance(start_d, str):
        start_d = str(start_d)
        tf.logging.fatal("start date must be in form 'now' or 'y.m.d.h.m'")

    if start_d == 'now':
        return datetime.now()
    return datetime(*map(lambda t: int(t), start_d.split('.')))


def find_training_data(start_date, train_period, train_data_path, pattern):
    """ Locate the data files for training.

    Args:
        start_date      (str): Current date
        train_period    (int): Number of previous days to use for training
        train_data_path (str): Data path for train TFrecords
        pattern         (str): Pattern for identifying Train/Eval/Test data

    Returns:
        data_file (list of str): List containing location of TFrecords
    """
    input_file_list = []
    now_time = parse_date(start_date)
    delta_time = timedelta(train_period)
    for f_name in os.listdir(train_data_path):
        if (pattern in f_name) and 'patch' not in f_name:
            f_time_str = re.search(re.compile(FILE_DATE_REGULAR_EXP), f_name).group()
            f_time = datetime.strptime(f_time_str, FILE_DATE_FORMAT)
            if now_time > f_time > now_time + delta_time:
                input_file_list.append(train_data_path + f_name)
    if input_file_list.__len__() < 1:
        tf.compat.v1.logging.fatal('File not exist under pattern "{0}", start date {1}, training period {2}'.format(
            pattern, start_date, train_period))
        return [-1]
    return input_file_list


def find_latest_model_dir(save_dir, store_dir, name):
    """ Parse model folder and find latest model parsed by time

    Args:
        save_dir  (str): Path to save models
        store_dir (str): Path to store models
        name      (str): Name of model, default set as 'dnn'

    Returns:
        model (str): model folder directory
    """
    path = save_dir + name + '_'
    directories = [i for i in glob(path + '*')]
    dir_date_list = [i.split(sep=path)[1] for i in directories]
    if len(dir_date_list) >= 1:
        model = name + '_' + max(dir_date_list)
    else:
        model = store_dir
    tf.compat.v1.logging.info('model: {0}_{1}'.format(name, model))
    return model


def write_new_training_to_csv(params):
    """ Write a new training to a csv file

    Args:
        params (dict): Dictionary of Hyper-parameters.
    """
    training_info = [params['store_dir'], 'Running', params['start_date'], params['train_period'],
                     params['is_test'], '--', '--', '--']

    with open(OUTPUT_CSV_FILE, 'a') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        # Write a header if file doesn't exist yet
        if os.stat(OUTPUT_CSV_FILE).st_size == 0:
            writer.writerow(CSV_FIELD_NAME)
        writer.writerow(training_info)


def update_training_status_to_csv(params, auc, ap, model_timestamp):
    """ Update training status to csv

    Args:
        params            (dict): Dictionary of Hyper-parameters.
        auc              (float): AUC
        ap               (float): Average precision
        model_timestamp    (int): Time Stamp created when exported saved_model
    """

    loader = csv.reader(open(OUTPUT_CSV_FILE))
    training_info = list(loader)
    for i in range(training_info.__len__()):
        if training_info[i][0] == params['store_dir']:
            training_info[i][1] = 'Finished'
            training_info[i][5] = auc
            training_info[i][6] = ap
            training_info[i][7] = model_timestamp
    writer = csv.writer(open(OUTPUT_CSV_FILE, 'w'))
    writer.writerows(training_info)


def write_error(params):
    """ Write error to csv file

    Args:
        params (dict): Dictionary of Hyper-parameters.
    """
    loader = csv.reader(open(OUTPUT_CSV_FILE))
    training_info = list(loader)
    for i in range(training_info.__len__()):
        if training_info[i][0] == params['store_dir']:
            training_info[i][1] = 'Error'
    writer = csv.writer(open(OUTPUT_CSV_FILE, 'w'))
    writer.writerows(training_info)


def find_data_size(file):
    """Find the size of TFrecords

    Args:
        file(list): Lists of file name
    Returns:
        count(int): dataset size
    """
    count = 0
    for fn in file:
        for _ in tf.python_io.tf_record_iterator(fn):
            count += 1
    return count


