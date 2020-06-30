import multiprocessing
import os

import tensorflow as tf

NUM_OF_CORES           = int(os.environ.get('NUM_OF_CORES') or multiprocessing.cpu_count())
OUTPUT_CSV_FILE        = os.environ.get('OUTPUT_CSV_FILE') or 'output_result.csv'
TRAIN_DATASET_PATTERN  = os.environ.get('TRAIN_DATASET_PATTERN') or 'train_data'
EVAL_DATASET_PATTERN   = os.environ.get('EVAL_DATASET_PATTERN') or 'valid_data'
TEST_DATASET_PATTERN   = os.environ.get('TEST_DATASET_PATTERN') or 'test_data'
FILE_DATE_FORMAT       = '%Y-%m-%dT%H:%M:%S'
FILE_DATE_REGULAR_EXP  = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d'

CSV_FIELD_NAME         = ['ModelName', 'TrainingStatus', 'StartDate', 'TrainingPeriod', 'IsTest', 'AUC', 'AP', 'Timestamp']
SHUFFLE_BUFFER_SIZE    = int(os.environ.get('SHUFFLE_BUFFER_SIZE') or 500)

# Categorical features = number of elements
NUM_WEEKDAY              = 8
NUM_REGION               = 396
NUM_CITY                 = 400
NUM_ADEXCHANGE           = 4
NUM_SLOTFORMAT           = 3
NUM_USERTAG              = 16707
NUM_CATEGORICAL_FEATURES = NUM_WEEKDAY+NUM_REGION + \
    NUM_CITY+NUM_ADEXCHANGE+NUM_SLOTFORMAT+NUM_USERTAG

MULTIHOT_FEAT_NUM_LIST = [NUM_USERTAG]
ONEHOT_FEAT_NUM_LIST   = [NUM_WEEKDAY,NUM_REGION,NUM_CITY,NUM_ADEXCHANGE,NUM_SLOTFORMAT]


categorical_features = ['weekday', 'region',
                        'city', 'adexchange', 'slotformat']
numerical_features = ['hour', 'slotwidth',
                      'slotheight', 'slotvisibility', 'slotprice']

RESPONSE_FEAT          = ['click']
INT_NUM_FEAT           = ['hour', 'slotwidth','slotheight', 'slotvisibility', 'slotprice']
INT_OHE_FEAT           = ['weekday', 'region','city', 'adexchange', 'slotformat']
INT_MHE_FEAT           = ['usertag']

FEATURE_LIST           = RESPONSE_FEAT + INT_NUM_FEAT + INT_OHE_FEAT +  INT_MHE_FEAT

# Create Feature Spec
FEATURE_SPEC = {}

for feature in RESPONSE_FEAT + INT_OHE_FEAT + INT_MHE_FEAT + INT_NUM_FEAT:
    FEATURE_SPEC.update({feature: tf.io.VarLenFeature(tf.int64)})
