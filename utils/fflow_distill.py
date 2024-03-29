import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time

sample_list = ['uniform', 'md', 'active']
agg_list = ['uniform', 'weighted_scale', 'weighted_com', 'none']
optimizer_list = ['SGD', 'Adam']


def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;',
                        type=str, default='mnist_cnum100_dist0_skew0_seed0')
    parser.add_argument(
        '--algorithm', help='name of algorithm;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;',
                        type=str, default='cnn')
    parser.add_argument(
        '--dropout_value', help='Value dropout in model;', type=float, default=0.2)

    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients',
                        type=str, choices=sample_list, default='uniform')
    parser.add_argument('--aggregate', help='methods for aggregating models',
                        type=str, choices=agg_list, default='uniform')
    parser.add_argument('--learning_rate_decay',
                        help='learning rate decay for the training process;', type=float, default=0.99)
    parser.add_argument(
        '--weight_decay', help='weight decay for the training process', type=float, default=0)
    parser.add_argument(
        '--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=0)
    # hyper-parameters of training in server side
    parser.add_argument(
        '--num_rounds', help='number of communication rounds', type=int, default=20)
    parser.add_argument(
        '--proportion', help='proportion of clients sampled per round', type=float, default=0.2)
    # hyper-parameters of local training
    parser.add_argument(
        '--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument(
        '--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument(
        '--batch_size', help='batch size when clients trainset on data;', type=int, default=64)
    parser.add_argument('--optimizer', help='select the optimizer for gd',
                        type=str, choices=optimizer_list, default='SGD')
    parser.add_argument(
        '--momentum', help='momentum of local update', type=float, default=0)

    # machine environment settings
    parser.add_argument(
        '--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--eval_interval',
                        help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument(
        '--num_threads', help='the number of threads;', type=int, default=1)
    parser.add_argument('--num_threads_per_gpu',
                        help="the number of threads per gpu in the clients computing session;", type=int, default=1)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--gpu', default=2, type=int)
    # the simulating system settings of clients

    # constructing the heterogeity of the network
    parser.add_argument(
        '--net_drop', help="controlling the dropout of clients after being selected in each communication round according to distribution Beta(drop,1)", type=float, default=0)
    parser.add_argument(
        '--net_active', help="controlling the probability of clients being active and obey distribution Beta(active,1)", type=float, default=99999)
    # constructing the heterogeity of computing capability
    parser.add_argument(
        '--capability', help="controlling the difference of local computing capability of each client", type=float, default=0)

    # hyper-parameters of different algorithms
    parser.add_argument('--learning_rate_lambda',
                        help='η for λ in afl', type=float, default=0)
    parser.add_argument('--q', help='q in q-fedavg', type=float, default='0.0')
    parser.add_argument('--epsilon', help='ε in fedmgda+',
                        type=float, default='0.0')
    parser.add_argument(
        '--eta', help='global learning rate in fedmgda+', type=float, default='1.0')
    parser.add_argument(
        '--tau', help='the length of recent history gradients to be contained in FedFAvg', type=int, default=0)
    parser.add_argument(
        '--alpha', help='proportion of clients keeping original direction in FedFV/alpha in fedFA', type=float, default='0.0')
    parser.add_argument('--beta', help='beta in FedFA',
                        type=float, default='1.0')
    parser.add_argument('--gamma', help='gamma in FedFA',
                        type=float, default='0')
    parser.add_argument('--mu', help='mu in fedprox',
                        type=float, default='0.2')

    parser.add_argument('--server_gpu_id',
                        help='server process on this gpu', type=int, default=0)

    parser.add_argument('--client_valid_ratio',
                        help='client_valid_ratio', type=float, default=0.25)

    # args for moving fed
    parser.add_argument(
        '--num_clients', help='Mean number of vehicle clients on the road', type=int, default=60)

    parser.add_argument(
        '--p_move', help='Probability of moving client', type=float, default=0.5)

    parser.add_argument(
        '--num_edges', help='Number of edge servers on the road', type=int, default=10)

    parser.add_argument('--edge_update_frequency',
                        help='Edge update frequency', type=int, default=1)
    parser.add_argument('--sample_with_replacement',
                        help='Sample with replacement or not', type=int, default=0)

    parser.add_argument('--distill_data_path',
                        help='path to save the distilled data', type=str, default=f'distill_data/')
    parser.add_argument(
        '--distill_ipc', help='Number of images per class to distill on each clients', type=int, default=1)
    parser.add_argument(
        '--distill_iters', help='Number of data distillation iterations', type=int, default=300)
    parser.add_argument(
        '--kip_support_size', help='Size of distill dataset if using KIP', type=int, default=50)
    parser.add_argument(
        '--remove_client', help='Setting remove client with setting', type=int, default=0)
    parser.add_argument(
        '--architec_KIP', help='select architecture for KIP method distill', default='FC')
    parser.add_argument(
        '--depth_KIP', help='select depth for KIP method distill', default=1, type=int)

    parser.add_argument('--distill_before_train',
                        help='Choose to distill before training or not', action='store_true', default=False)

    try:
        option = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    return option


def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(123+seed)


def initialize(option):
    # init fedtask
    # init fedtask
    print("init fedtask...", end='')
    # dynamical initializing the configuration with the benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(
        ['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:{}'.format(
        option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
    utils.fmodule.TaskCalculator = getattr(
        importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(
        getattr(importlib.import_module('torch.optim'), option['optimizer']))
    utils.fmodule.Model = getattr(
        importlib.import_module(bmk_model_path), 'Model')
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(
        taskpath=os.path.join('fedtask', option['task']))
    print(task_reader)

    print('Reading task_reader')
    train_datas, valid_datas, test_data, client_names = task_reader.read_data()

    print("Training data")
    print(train_datas[0].get_all_labels())
    num_clients = len(client_names)
    print("done")

    # init client
    print('init clients...', end='')
    client_path = '%s.%s' % ('algorithm.distill_fl', option['algorithm'])
    Client = getattr(importlib.import_module(client_path), 'MobileClient')
    clients = [Client(option, name=client_names[cid], train_data=train_datas[cid],
                      valid_data=valid_datas[cid]) for cid in range(num_clients)]
    print('done')

    # print("Clients", clients)

    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('algorithm.distill_fl', option['algorithm'])
    server = getattr(importlib.import_module(server_path), 'CloudServer')(
        option, utils.fmodule.Model().to(utils.fmodule.device), clients=clients, test_data=test_data)
    print('done')
    return server    # print('done')


def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para in server.paras_name:
        header = header + para + "{}_".format(option[para])
    output_name = header + "M{}_R{}_B{}_E{}_LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_DR{:.2f}_AC{:.2f}.json".format(
        option['model'],
        option['num_rounds'],
        option['batch_size'],
        option['num_epochs'],
        option['learning_rate'],
        option['proportion'],
        option['seed'],
        option['lr_scheduler']+option['learning_rate_decay'],
        option['weight_decay'],
        option['net_drop'],
        option['net_active'])
    return output_name


class Logger:
    def __init__(self):
        self.output = {}
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf = {}

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key=''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key=''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event."""
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] = time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(
                key+":", self.time_buf[key][-1]) + 's')

    def save(self, filepath):
        """Save the self.output as .json file"""
        if self.output == {}:
            return
        with open(filepath, 'w') as outf:
            ujson.dump(self.output, outf)

    def write(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name == None:
            raise RuntimeError(
                "Missing the name of the variable to be logged.")
        if var_name in [key for key in self.output.keys()]:
            self.output[var_name] = []
        self.output[var_name].append(var_value)
        return

    def log(self, server=None):
        pass
