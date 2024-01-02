import logging
import datetime
from algorithm.distill_fl.rfad.rfad_distillation import RFAD_Distillation
from multiprocessing import Pool as ThreadPool
from tqdm import tqdm
import os
from main_distill import logger
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import math
import copy
from benchmark.toolkits import XYDataset
from .fedbase_mobile_distill import BasicCloudServer, BasicEdge, BasicMobileClient
import random

import torch
from utils import fmodule
import sys
sys.path.append('..')


now = datetime.datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

directory_path = "log"  # specify the directory where log files will be saved
log_file_name = f'{directory_path}/log_kip_fedbase_{formatted_date_time}.log'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a file handler
file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.DEBUG)

# Create a stream handler (for console output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger_anhtn = logging.getLogger('')
logger_anhtn.addHandler(file_handler)
logger_anhtn.addHandler(stream_handler)


class CloudServer(BasicCloudServer):
    def __init__(self, option, model, clients, test_data=None):
        super(CloudServer, self).__init__(option, model, clients, test_data)
        self.initialize()

        self.avg_edge_train_losses = []
        self.avg_edge_valid_losses = []
        self.avg_edge_train_metrics = []
        self.avg_edge_valid_metrics = []
        self.edge_client_communication_cost = []
        self.edge_cloud_communication_cost = []
        self.edge_metrics = {}

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # First, distill all data on clients' side
        # for client in self.clients:
        #     client.distill_data()

        num_iterations_start_remove = 50
        num_clients_removed = 2
        after_num_itr_remove = 5
        if self.option['remove_client'] == 1:
            if t >= num_iterations_start_remove and t % after_num_itr_remove == 0:
                self.delete_clients(num_clients_removed)

        # # sample clients: MD sampling as default but with replacement=False
        # # #print("Iterating")
        # self.global_update_location()
        # # #print("Done updating location")
        # self.update_client_list()
        # # #print("Done updating client_list")
        # self.assign_client_to_server()
        # # #print("Done assigning client to sercer")
        # #print("client name: ")
        # #print([client.name for client in  self.clients])

        # print(self.client_edge_mapping)
        self.update_client_server_mapping()
        # #print(self.client_edge_mapping)
        # #print(self.edge_client_mapping)

        for client in self.clients:
            if client.name == 'Client000':
                print(client.current_edge_name)

        for edge in self.edges:
            if 'Client000' in self.edge_client_mapping[edge.name]:
                print(edge.name)

        self.selected_clients = self.sample()
        # print("Selected clients", len(self.selected_clients))
        # print("Transfer data of client selected", len(self.selected_clients))

        # first, aggregate the edges with their clientss
        # for client in self.selected_clients:
        #     client.print_client_info()
        all_total_transfer_size = []
        for edge in self.edges:
            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in self.edge_client_mapping[edge.name]:
                    aggregated_clients.append(client)
            # print("Agg clients" ,aggregated_clients)

            if len(aggregated_clients) > 0:
                edge.collect_distilled_data_from_client(aggregated_clients)
                all_total_transfer_size.append(edge.total_transfer_size)

        models, (edge_names, train_losses, valid_losses, train_acc,
                 valid_acc) = self.communicate(self.edges)
        all_edge_train_losses = []
        all_edge_valid_losses = []
        all_edge_train_metrics = []
        all_edge_valid_metrics = []

        for i in range(len(edge_names)):
            edge_name = edge_names[i]
            edge_train_loss = train_losses[i]
            edge_valid_loss = valid_losses[i]
            edge_train_acc = train_acc[i]
            edge_valid_acc = valid_acc[i]
            if edge_name in self.edge_metrics.keys():
                self.edge_metrics[edge_name].append(
                    [t, edge_train_loss, edge_train_acc, edge_valid_loss, edge_valid_acc])
            else:
                self.edge_metrics[edge_name] = [
                    ['Round', 'train_losses', 'train_accs', 'val_Losses', 'val_accs']]

            all_edge_train_losses.append(edge_train_loss)
            all_edge_valid_losses.append(edge_valid_loss)
            all_edge_train_metrics.append(edge_train_acc)
            all_edge_valid_metrics.append(edge_valid_acc)

            # else:
            #     print('No aggregated clients')
        # models, train_losses = self.communicate(self.edges)

        # print("Done a training step")
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients:
            return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        # models = [edge.model for edge in self.edges]
        communication_edge_to_cloud = 0
        if t % self.edge_update_frequency == 0:
            models = [edge.model for edge in self.edges]

            total_size_edge_1side_cloud = sum(
                self.get_model_size(model) for model in models)
            # !nhân 2 vì có cả chiều đi và chiều về
            communication_edge_to_cloud = total_size_edge_1side_cloud * 2

            sum_datavol = sum([edge.datavol for edge in self.edges])
            edge_weights = [edge.datavol / sum_datavol for edge in self.edges]
            self.model = self.aggregate(models, p=edge_weights)

            for edge in self.edges:
                edge.model = copy.deepcopy(self.model)

        self.edge_cloud_communication_cost.append(communication_edge_to_cloud)

        self.avg_edge_train_losses.append(
            sum(all_edge_train_losses) / len(all_edge_train_losses))
        self.avg_edge_valid_losses.append(
            sum(all_edge_valid_losses) / len(all_edge_valid_losses))
        self.avg_edge_train_metrics.append(
            sum(all_edge_train_metrics) / len(all_edge_train_metrics))
        self.avg_edge_valid_metrics.append(
            sum(all_edge_valid_metrics) / len(all_edge_valid_metrics))
        self.edge_client_communication_cost.append(
            sum(all_total_transfer_size))

    # def sample(self):
    #     """Sample the clients.
    #     :param
    #         replacement: sample with replacement or not
    #     :return
    #         a list of the ids of the selected clients
    #     """
    #     # print("Sampling selected clients")
    #     all_clients = [cid for cid in range(self.num_clients)]
    #     # print("Done all clients")
    #     selected_clients = []
    #     # collect all the active clients at this round and wait for at least one client is active and
    #     active_clients = []
    #     active_clients = self.clients
    #     # while(len(active_clients)<1):
    #     #     active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
    #     # print("DOne collect all the active clients")
    #     # sample clients
    #     if self.sample_option == 'active':
    #         # select all the active clients without sampling
    #         selected_clients = active_clients
    #     if self.sample_option == 'uniform':
    #         # original sample proposed by fedavg
    #         selected_clients = list(np.random.choice(active_clients, self.clients_per_round, replace=False))
    #     elif self.sample_option =='md':
    #         # the default setting that is introduced by FedProx
    #         selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
    #     # drop the selected but inactive clients
    #     selected_clients = list(set(active_clients).intersection(selected_clients))
    #     return selected_clients

    def initialize_edges(self):
        name_lists = ['e' + str(client_id)
                      for client_id in range(self.num_edges)]
        self.edges = []
        self.edges_names = []
        for i in range(self.num_edges):
            edge = EdgeServer(self.option, model=copy.deepcopy(
                self.model), name=name_lists[i], test_data=None)
            edge_name = name_lists[i]
            self.edges.append(edge)
            self.edges_names.append(edge_name)
            self.edge_to_name_mapping[edge] = edge_name
            self.name_to_edge_mapping[edge_name] = edge

    def print_clients_info(self):
        print("Current number of clients: ", self.current_num_clients)
        for client in self.clients:
            client.print_client_info()

    # def initialize(self):
    #     self.initialize_edges()
    #     self.assign_client_to_server()
    #     self.initialize_clients_location_velocity()

    def initialize(self):
        self.initialize_edges()
        self.initialize_clients()
        self.initialize_client_to_server()


class EdgeServer(BasicEdge):
    def __init__(self, option, model, name='', clients=[], test_data=None):
        super(EdgeServer, self).__init__(
            option, model, name, clients, test_data)
        self.clients = []

    def update_client_list(self, clients):
        import pdb
        pdb.set_trace()
        self.clients = clients

    def print_edge_info(self):
        print('Edge {} - cover area: {}'.format(self.name, self.cover_area))


class MobileClient(BasicMobileClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(
            option,   name, train_data, valid_data)
        # self.velocity = velocity
        self.associated_server = None

        if 'mnist' in self.option['task'] or 'cifar10' in self.option['task']:
            self.num_classes = 10
        elif 'cifar100' in self.option['task']:
            self.num_classes = 100
        self.option = option
        self.ipc = self.option['distill_ipc']
        self.support_size = self.option['kip_support_size']
        self.distill_iters = self.option['distill_iters']
        self.task_name = self.option['task']
        self.distill_save_path = os.path.join(
            f'fedtask/{self.task_name}/', self.option['distill_data_path'], str(self.option['distill_ipc']))
        self.total_size = 0
        if not os.path.exists(self.distill_save_path):
            os.makedirs(self.distill_save_path)
        self.distill_save_path = os.path.join(
            self.distill_save_path, f'{self.name}/')
        # #print("Path", self.distill_save_path)
        # self.distill_save_path = os.path.join(self.distill_save_path, self.option['task'])
        if not os.path.exists(self.distill_save_path):
            os.mkdir(self.distill_save_path)

        if 'mnist' in self.option['task']:
            self.dataset = 'MNIST'
        elif 'cifar10' in self.option['task']:
            self.dataset = 'CIFAR10'
        elif 'cifar100' in self.option['task']:
            self.dataset = 'CIFAR100'

        self.distill_options = {
            'coreset': False, 'corruption': 0, 'dataset': 'cifar10', 'ga_steps': 1,
            'init_strategy': 'random', 'jit': 0.005, 'learn_labels': True, 'lr': 0.001,
            'n_batches': 4, 'n_models': 8, 'path_dataset': '../data', 'platt': True,
            'samples_per_class': 10, 'save_path': './result', 'seed': 0
        }

        self.distiller = RFAD_Distillation(server_gpu_id=self.option['server_gpu_id'],
                                           n_iters=self.distill_iters,
                                           dataset=self.dataset.lower(), save_path=self.distill_save_path,
                                           ipc=self.ipc)

    def distill_data(self):

        message = f"Distilling data from client: {self.name}"
        print(message)
        # import pdb; pdb.set_trace()
        x_train, y_train, x_val, y_val = self.train_data.X, self.train_data.Y, self.valid_data.X, self.valid_data.Y
        print(f'Client name: {self.name}')
        print(f'Check data class for each client. Client: {self.name}')
        print(set(y_train))
        logger_anhtn.info(
            f'Check data class for each client. Client: {self.name}')
        logger_anhtn.info(set(y_train))
        # print("Data from client"x_val,y_val)
        self.distiller.distill(X_TRAIN_RAW=x_train, LABELS_TRAIN=y_train, X_TEST_RAW=x_val,
                               LABELS_TEST=y_val,  options=self.distill_options)

    def print_client_info(self):
        # print('Client {} - current loc: {} - velocity: {} - training data size: {}'.format(self.name,self.location,self.velocity,
        #
        pass

    def get_size_of_data(self, path_get_size, message='None'):
        file_stats = os.stat(path_get_size)
        bytes_size = file_stats.st_size
        # print(f'{self.name} - {message} - {bytes_size} Bytes')
        return file_stats.st_size

    def calculate_total_size(self):
        file_path_x = os.path.join(self.distill_save_path, 'x_distill.pt')
        file_path_y = os.path.join(self.distill_save_path, 'y_distill.pt')
        size_of_x = self.get_size_of_data(file_path_x, 'x')
        size_of_y = self.get_size_of_data(file_path_y, 'y')
        self.total_size = size_of_x + size_of_y

    def load_distill_data(self):
        self.calculate_total_size()
        self.x_distill = torch.load(os.path.join(
            self.distill_save_path, 'x_distill.pt'))
        self.y_distill = torch.load(os.path.join(
            self.distill_save_path, 'y_distill.pt'))
        # self.y_distill = torch.Tensor(self.y_distill)
        self.x_distill = np.array(self.x_distill.detach().cpu())

        # print(self.x_distill,type(self.x_distill))
        # print(self.y_distill, type(self.y_distill))

        # print(self.x_distill, self.y_distill)
