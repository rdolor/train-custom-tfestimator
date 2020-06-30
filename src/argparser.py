from configargparse import ArgParser
from src.config import FILE_DATE_FORMAT, INT_NUM_FEAT
from src.utils import parse_date, find_latest_model_dir

import tensorflow as tf
import os

def parse_args():
    parser = ArgParser(default_config_files=[os.getcwd() + '/src/initial_configurations/default'])
    # Core setting
    core_parse = parser.add_argument_group('Core setting')
    core_parse.add_argument('-s',   '--start_date',   dest='start_date',   default='now', type=str, help='Training start date')
    core_parse.add_argument('-p',   '--train_period', dest='train_period', default=-1,    type=int, help='Time period of training file is used')
    core_parse.add_argument('-n',   '--new_run',      dest='new_run',      default=0,     type=int, help='If the model checkpoint is erased to run new model')
    core_parse.add_argument('-l',   '--local_run',    dest='local_run',    default=0,     type=int, help='If the parameter JSON file is kept locally insteat to  redis')
    core_parse.add_argument('-nni', '--tuning',       dest='tuning',       default=0,     type=int, help='Whether or not to peform NNI hyper parameter tuning')


    # Model
    model_parse = parser.add_argument_group('Model')
    model_parse.add_argument('-m',   '--model',   dest='model',           default='DNN',               type=str,   help='Select the model to train e.g. DNN')
    model_parse.add_argument('--loss',            dest='loss',            default=30,                  type=int,   help="Setting of loss function '10','11','12','20','21','22','30','31','32'" )
    model_parse.add_argument('--hidden_units',    dest='hidden_units',    default=[128, 64],           type=int,   nargs='+', help='List containing the number of hidden units to use for each hidden layer')
    model_parse.add_argument('--dropout_rate',    dest='dropout_rate',    default=0.5,                 type=float, help='List containing the number of dropout rate to use for each hidden layer')
    model_parse.add_argument('--one_hot_units',   dest='one_hot_units',   default=[2, 35, 359, 3, 2], type=int,   nargs='+', help='List containing the number of embedding units to use for features (in order): [weekday, region, city, adexchange, slotformat]; this replaces the one hot encoding')
    model_parse.add_argument('--multi_hot_units', dest='multi_hot_units', default=[45],             type=int,   nargs='+', help='List containing the number of embedding units to use for features: [usertag]')
    model_parse.add_argument('--learning_rate',   dest='learning_rate',   default=0.002,            type=float, help='Learning rate of updating gradient')
    model_parse.add_argument('--decay_step',      dest='decay_step',      default=100,              type=int,   help='Decay step')
    model_parse.add_argument('--decay_rate',      dest='decay_rate',      default=0.98,             type=float, help='Decay rate for exponential decay of learning rate')
    model_parse.add_argument('--class_ratio',     dest='class_ratio',     default=0.5,              type=float, help='Ratio of 2 classes for imbalanced data')
    model_parse.add_argument('--alpha',           dest='alpha',           default=1.,               type=float, help='Alpha for Focal loss regularization in DNN')
    model_parse.add_argument('--beta',            dest='beta',            default=1.,               type=float, help='Beta for regularization')
    model_parse.add_argument('--gamma',           dest='gamma',           default=1.,               type=float, help='Gamma for Focal loss regularization in DNN')

    # Training
    train_parse = parser.add_argument_group('Training hyperparameters')
    train_parse.add_argument('--save_summary_steps',   dest='save_summary_steps',    default=100,   type=int, help='save summary steps')
    train_parse.add_argument('--log_step_count_steps', dest='log_step_count_steps',  default=100,   type=int, help='logging step count steps')
    train_parse.add_argument('--checkpoints_steps',    dest='save_checkpoints_steps',default=500,   type=int, help='checkpoints steps')
    train_parse.add_argument('--has_gpu',              dest='has_gpu',               default=0,     type=int, help='1 if GPU is present, else 0')
    train_parse.add_argument('--oversample',           dest='oversample',            default=0,     type=int, help='1 if will oversample training dataset, else 0')
    train_parse.add_argument('--is_test',              dest='is_test',               default=0,     type=int, help='1 if the trained model will be evaluated, else 0')
    train_parse.add_argument('--num_epochs',           dest='num_epochs',            default=1.0,   type=float, help='Number of total epochs')
    train_parse.add_argument('--start_delay_secs',     dest='start_delay_secs',      default=10,    type=int, help='Start evaluating after 10 secs')
    train_parse.add_argument('--throttle_secs',        dest='throttle_secs',         default=10,    type=int, help='Evaluate only every 30 secs')
    train_parse.add_argument('--batch_size',           dest='batch_size',            default=128,      type=int, help='Number of examples per batch')
    
    # Directory paths
    dir_parse = parser.add_argument_group('Directory paths')
    dir_parse.add_argument('--train_data_path',  dest='train_data_path',  default='./data/',        type=str, help='Directory where the training files are located')
    dir_parse.add_argument('--save_dir',         dest='save_dir',         default='./Outputs/',    type=str, help='Directory to save model directories')
    dir_parse.add_argument('--load_dir',         dest='load_dir',         default='latest',        type=str, help='Directory to load old model,default "new" as the latest model')
    dir_parse.add_argument('--store_dir',        dest='store_dir',        default='latest',        type=str, help='Directory to store current model, default "latest" to save in timestamp')
    dir_parse.add_argument('--builder_save_dir', dest='builder_save_dir', default='builder_save',  type=str, help='Directory to store current model for tfjs predictor')

    _args, _ = parser.parse_known_args()
    _params = vars(_args)
    _params['train_data_path'] = os.getcwd() + _params['train_data_path']

    # Identify whether it's using NNI tuning mode
    if _params['tuning'] == 1:
        import nni
        tuner_params = nni.get_next_parameter()
        try:
            _params.update(tuner_params)
        except Exception as err:
            tf.logging.error('Error args updated: %s', err)
            tf.logging.error('Failed with params: %s', str(_params))
            
    _params['num_features'] = len(INT_NUM_FEAT) + sum(_params['one_hot_units']) + sum(_params['multi_hot_units'])
    _params['model_name'] = _params['model']

    # Adjust filename to restore/save by config settings
    if _params['store_dir'] == 'latest':
        _params['store_dir'] = _params['model_name'] + '_' + parse_date('now').strftime(FILE_DATE_FORMAT)
    if _params['load_dir'] == 'latest':
        _params['load_dir'] = find_latest_model_dir(_params['save_dir'], _params['store_dir'], _params['model_name'])
    if _params['new_run'] == 1:
        _params['load_dir'] = _params['store_dir']
    return _params
