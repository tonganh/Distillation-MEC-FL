import random

import torch
from utils import fmodule
import sys
sys.path.append('..')
from .fedbase_mobile  import BasicCloudServer, BasicEdgeServer, BasicMobileClient
from benchmark.toolkits import XYDataset
import copy
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from main_mobile import logger
import os
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool
# from .mobile_fl_utils import NTD_Loss, model_weight_divergence, kl_divergence, calculate_kl_div_from_data, SoftTargetDistillLoss
import torch.nn as nn


class CloudServer(BasicCloudServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(CloudServer, self).__init__( option, model,clients,test_data )
        self.initialize()


    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')
            # print(self.clients)
            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)

        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        # logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        
        self.selected_clients = self.clients
        print("Selected clients", len(self.selected_clients))
        num_iterations_start_remove = 50
        num_clients_removed = 2
        after_num_itr_remove = 5
        if self.option['remove_client'] == 1:
            if t >= num_iterations_start_remove and t%after_num_itr_remove==0:
                self.delete_clients(num_clients_removed)

        self.selected_clients = self.clients

        all_client_train_losses = []
        all_client_valid_losses = []
        all_client_train_metrics = []
        all_client_valid_metrics = []

        communication_cost_in_round = 0
        trained_clients = []

        trained_clients = []

        for edge in self.edges:
            np.random.seed(self.option['seed'] +21)

            clients_chosen_in_edge =     list(np.random.choice(self.edge_client_mapping[edge.name],
                                                               int(len(self.edge_client_mapping[edge.name]) * self.option['proportion']), replace=False))
            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in clients_chosen_in_edge:
                    aggregated_clients.append(client)
                    trained_clients.append(client)

            if len(aggregated_clients) > 0:
                # print(aggregated_clients)
                # print(edge.communicate(aggregated_clients))
                global_model  =copy.deepcopy(self.model)
                aggregated_clients_models , (agg_clients_train_losses, 
                                             agg_clients_valid_losses, 
                                             agg_clients_train_accs, 
                                             agg_clients_valid_accs)= edge.communicate(aggregated_clients,global_model )
                
                # edge_total_datavol = sum([client.datavol for client in aggregated_clients])
                # edge.total_datavol = edge_total_datavol
                # aggregation_weights = [client.datavol / edge_total_datavol for client in aggregated_clients]
                # # print(len(aggregation_weights), len(aggregated_clients_models))
                # edge.model =  self.aggregate(aggregated_clients_models, p = aggregation_weights)
                # edge.add_model_to_buffer()
                # edge.get_ensemble_model()

                all_client_train_losses.extend(agg_clients_train_losses)
                all_client_valid_losses.extend(agg_clients_valid_losses)
                all_client_train_metrics.extend(agg_clients_train_accs)
                all_client_valid_metrics.extend(agg_clients_valid_accs)
        
        self.client_train_losses.append(sum(all_client_train_losses) / len(all_client_train_losses))
        self.client_valid_losses.append(sum(all_client_valid_losses) / len(all_client_valid_losses))
        self.client_train_metrics.append(sum(all_client_train_metrics) / len(all_client_train_metrics))
        self.client_valid_metrics.append(sum(all_client_valid_metrics) / len(all_client_valid_metrics))

        self.global_update_location()
        self.update_client_list()
        self.assign_client_to_server()

        for edge in self.edges:

            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in self.client_edge_mapping[edge.name]:
                    if client in trained_clients:
                        # print(edge.name,client.name)
                        aggregated_clients.append(client)
    
            if len(aggregated_clients) > 0:
                # print('Chosen')
                aggregated_clients_models = [client.model for client in aggregated_clients]
                edge_total_datavol = sum([client.datavol for client in aggregated_clients])
                edge.total_datavol = edge_total_datavol
                aggregation_weights = [client.datavol / edge_total_datavol for client in aggregated_clients]
                edge.model =  self.aggregate(aggregated_clients_models, p = aggregation_weights)
                edge.add_model_to_buffer()
                edge.get_ensemble_model()

        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        # models = [edge.model for edge in self.edges]
        if t % self.edge_update_frequency == 0:
            models = [edge.model for edge in self.edges]
            sum_datavol = sum([edge.total_datavol for edge in self.edges])
            edge_weights = [edge.total_datavol / sum_datavol for edge in self.edges]
            self.model = self.aggregate(models, p = edge_weights)

            for edge in self.edges:
                edge.model = copy.deepcopy(self.model)
                edge.reset_buffer()

        edges_models_list = []
        for edge in self.edges:
                edges_models_list.append(copy.deepcopy(edge.model))
     
        


    def communicate(self, edges):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_edges = []
        if self.num_threads <= 1:
            # computing iteratively
            for edge in edges:
                response_from_edge = self.communicate_with(edge)
                packages_received_from_edges.append(response_from_edge)
    
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(edges)))
            packages_received_from_edges = pool.map(self.communicate_with, edges)
            pool.close()
            pool.join()
        # count the clients not dropping
        # self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        # packages_received_from_edges = [pi for pi in packages_received_from_clients if pi]
        return self.unpack(packages_received_from_edges)

    def communicate_with(self, edge):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client: the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # package the necessary information
        svr_pkg = self.pack()

        # listen for the client's response and return None if the client drops out
        # if self.clients[client_id].is_drop(): return None
        reply = edge.reply(svr_pkg)
        return reply

    def pack(self):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "model" : copy.deepcopy(self.model),
        }

    def sample(self):
        """Sample the clients.
        :param
            replacement: sample with replacement or not
        :return
            a list of the ids of the selected clients
        """
        # print("Sampling selected clients")
        all_clients = [cid for cid in range(self.num_clients)]
        # print("Done all clients")
        selected_clients = []
        # collect all the active clients at this round and wait for at least one client is active and
        active_clients = []
        active_clients = self.clients
        # while(len(active_clients)<1):
        #     active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
        # print("DOne collect all the active clients")
        # sample clients
        if self.sample_option == 'active':
            # select all the active clients without sampling
            selected_clients = active_clients
        if self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(active_clients, self.clients_per_round, replace=False))
        elif self.sample_option =='md':
            # the default setting that is introduced by FedProx
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
        # drop the selected but inactive clients
        selected_clients = list(set(active_clients).intersection(selected_clients))
        return selected_clients



    # def create_clients(self, num_clients):
    #     locations = np.random.randint( self.left_road_limit, self.right_road_limit, size = num_clients)
    #     if self.option['mean_velocity'] != 0:
    #         velocities = np.random.randint( - (self.mean_velocity + self.std_velocity), self.mean_velocity + self.std_velocity, size = num_clients)
    #     else:
    #         velocities = np.array([0 for i in range(num_clients)])
    #     name_lists = ['c' + str(client_id) for client_id in range(num_clients)]
    #     if self.option['sample_with_replacement']:
    #         client_data_lists = self.sample_data_with_replacement(num_clients=num_clients)
    #     else:
    #         client_data_lists = self.sample_data_without_replacement(num_clients=num_clients)

    #     new_clients = []
    #     for i in range(num_clients):
    #         client_train_data, client_valid_data = client_data_lists[i]
    #         client = MobileClient(self.option, location=locations[i], velocity=velocities[i], name=name_lists[i], 
    #                               train_data=client_train_data, valid_data = client_valid_data)
    #         new_clients.append(client)
        
    #     return new_clients
    
    # def initialize_clients(self):
    #     self.clients = self.create_clients(self.current_num_clients)
    
        
    def initialize_edges(self):
        name_lists = ['e' + str(client_id) for client_id in range(self.num_edges)]
        self.edges = []
        self.edges_names = []
        for i in range(self.num_edges):
            edge = BasicEdgeServer(self.option, model = copy.deepcopy(self.model), name=name_lists[i], test_data = None)
            edge_name = name_lists[i]
            self.edges.append(edge)
            self.edges_names.append(edge_name)
            self.edge_to_name_mapping[edge] = edge_name
            self.name_to_edge_mapping[edge_name] = edge

    def print_clients_info(self):
        print("Current number of clients: ", self.current_num_clients)
        for client in self.clients:
            client.print_client_info()

    def initialize(self):
        self.initialize_edges()
        self.initialize_clients()
        self.initialize_client_to_server()


class EdgeServer(BasicEdgeServer):
    def __init__(self, option,model, name = '', clients = [], test_data=None):
        super(EdgeServer, self).__init__(option,model, name , clients , test_data)
        self.clients = []
        self.model_buffer = []
        self.max_buffer_size = option['edge_model_buffer_size'] 
        self.ensemble_model = None

    def update_client_list(self,clients):
        self.clients = clients
    
    def get_data(self):
        all_edge_data = []
        for client in self.clients:
            # print(client.train_data.X.shape)
            # return client.train_data.X
            all_edge_data.append(client.train_data.X)
        
        edge_data = torch.cat(all_edge_data,0)
        # print(edge_data.shape)
        return edge_data
    
    def add_model_to_buffer(self):
        self.model_buffer.append(self.model)
        
        if len(self.model_buffer) > self.max_buffer_size:
            self.model_buffer = self.model_buffer[-self.max_buffer_size:]
    
    def reset_buffer(self):
        self.model_buffer = []

    def get_ensemble_model(self):
        # print(len(self.model_buffer))
        self.ensemble_model = fmodule._model_average(self.model_buffer)
    

    # def get_client_distribution()

    def print_edge_info(self):
        print('Edge {} - cover area: {}'.format(self.name,self.cover_area))

    def communicate(self, clients, global_model):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        if self.num_threads <= 1:
            # computing iteratively
            for client in clients:
                response_from_edge = self.communicate_with(client, global_model)
                packages_received_from_clients.append(response_from_edge)
    
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(clients)))
            packages_received_from_clients = pool.map(self.communicate_with,clients)
            pool.close()
            pool.join()
        # count the clients not dropping
        # self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        # packages_received_from_edges = [pi for pi in packages_received_from_clients if pi]
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client, global_model):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client: the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # package the necessary information
        edge_pkg = self.pack(global_model)
        # listen for the client's response and return None if the client drops out
        # if self.clients[client_id].is_drop(): return None
        reply = client.reply(edge_pkg)
        return reply

    def pack(self, global_model):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "edge_model" : copy.deepcopy(self.model),
            'edge_ensemble_model': copy.deepcopy(self.ensemble_model),
            "global_model" : copy.deepcopy(global_model),
        }

    def unpack_svr(self, received_pkg):
        """
        Unpack the package received from the cloud server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model']


class MobileClient(BasicMobileClient):
    def __init__(self, option,  name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(option, name, train_data, valid_data)
        # self.velocity = velocity
        self.associated_server = None
        self.global_mu = option['mu_global']
        self.edge_mu = option['mu_edge']

        # self.T  = option['distill_temperature']
        self.model = None

        # self.distill_loss = SoftTargetDistillLoss(self.T)

        if 'cifar100' in option['task']:
            num_classes = 100
        else:
            num_classes = 10
        self.alpha = option['distill_alpha']
        self.option = option
        # self.distill_loss = NTD_Loss(num_classes=num_classes, tau = self.T)


        # self.global_beta = option['distill_global_beta']

    def print_client_info(self):
        print('Client {} - current loc: {} - velocity: {} - training data size: {}'.format(self.name,self.location,self.velocity,
                                                                                           self.datavol))

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['edge_model'],  received_pkg['edge_ensemble_model'], received_pkg['global_model']

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        # print("In reply function of client")
        edge_model, edge_ensemble_model, global_model = self.unpack(svr_pkg)
        # print("CLient unpacked to package")
        train_loss = self.train_loss(edge_model)
        valid_loss = self.valid_loss(edge_model)
        train_acc = self.train_metrics(edge_model)
        valid_acc = self.valid_metrics(edge_model)

        # print("Client evaluated the train losss")
        self.train(edge_model,edge_ensemble_model, global_model)
        # print("Client trained the model")
        eval_dict = {'train_loss': train_loss, 
                      'valid_loss': valid_loss,
                      'train_acc':train_acc,
                      'valid_acc': valid_acc}
        cpkg = self.pack(edge_model, eval_dict)
        # print("Client packed and finished")
        return cpkg

    def train(self, model, edge_ensemble_model, global_model):
        # print(device)
        # global parameters
        edge_model = copy.deepcopy(model)
        global_model = copy.deepcopy(global_model)
        edge_model.freeze_grad()
        # if global_teacher != None:
        global_model.freeze_grad()

        model.train()

        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, edge_model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
            
                model.zero_grad()
                original_loss = self.calculator.get_loss(model, batch_data)
                # proximal term
                edge_proximal = 0
                for pm, ps in zip(model.parameters(), edge_model.parameters()):
                    edge_proximal += torch.sum(torch.pow(pm-ps,2))
                
                global_proximal = 0
                for pm, ps in zip(model.parameters(), global_model.parameters()):
                    global_proximal += torch.sum(torch.pow(pm-ps,2))

                loss = original_loss + 0.5 * self.edge_mu * edge_proximal    + 0.5 * self.global_mu * global_proximal             #
                loss.backward()
                optimizer.step()

        self.model = copy.deepcopy(edge_model)

        return
        