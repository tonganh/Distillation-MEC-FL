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
# from .mobile_fl_utils import NTD_Loss, model_weight_divergence, kl_divergence, calculate_kl_div_from_data
from .mobile_fl_utils import SoftTargetDistillLoss

class CloudServer(BasicCloudServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(CloudServer, self).__init__( option, model,clients,test_data )
        self.initialize()    
        print("Init", self.client_edge_mapping)

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        print("Run", self.client_edge_mapping)

        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')
            # #print(self.clients)
            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)

        print("=================End==================")
        logger.time_end('Total Time Cost')


    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # print("iterate", self.client_edge_mapping)

        self.selected_clients = self.clients
        # print("Selected clients", len(self.selected_clients))
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
                total_size_edge_1side_cloud = sum(self.get_model_size(model) * 2 for model in aggregated_clients_models)
                # !nhân 2 vì có cả chiều đi và chiều về
                communication_cost_in_round += total_size_edge_1side_cloud*2


                all_client_train_losses.extend(agg_clients_train_losses)
                all_client_valid_losses.extend(agg_clients_valid_losses)
                all_client_train_metrics.extend(agg_clients_train_accs)
                all_client_valid_metrics.extend(agg_clients_valid_accs)
        
        self.client_train_losses.append(sum(all_client_train_losses) / len(all_client_train_losses))
        self.client_valid_losses.append(sum(all_client_valid_losses) / len(all_client_valid_losses))
        self.client_train_metrics.append(sum(all_client_train_metrics) / len(all_client_train_metrics))
        self.client_valid_metrics.append(sum(all_client_valid_metrics) / len(all_client_valid_metrics))

        # self.global_update_location()
        # self.update_client_list()
        # self.assign_client_to_server()

        self.update_client_server_mapping()
        # print(self.selected_clients)

        for edge in self.edges:

            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in self.edge_client_mapping[edge.name]:
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


                # edge.add_model_to_buffer()
                # edge.get_ensemble_model()

        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        # models = [edge.model for edge in self.edges]
        communication_cost_edge_cloud = 0

        if t % self.edge_update_frequency == 0:
            models = [edge.model for edge in self.edges]
            total_size_edge_1side_cloud = sum(self.get_model_size(model) for model in models)
            communication_cost_edge_cloud = total_size_edge_1side_cloud*2

            sum_datavol = sum([edge.total_datavol for edge in self.edges])
            edge_weights = [edge.total_datavol / sum_datavol for edge in self.edges]
            self.model = self.aggregate(models, p = edge_weights)

            for edge in self.edges:
                edge.model = copy.deepcopy(self.model)
                edge.reset_buffer()

        edges_models_list = []
        for edge in self.edges:
                edges_models_list.append(copy.deepcopy(edge.model))
     
        self.edge_to_cloud_communication_cost.append(communication_cost_edge_cloud)
        self.client_to_edge_communication_cost.append(communication_cost_in_round)

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

    def initialize_edges(self):
        name_lists = ['e' + str(client_id) for client_id in range(self.num_edges)]
        self.edges = []
        self.edges_names = []
        for i in range(self.num_edges):
            edge = EdgeServer(self.option, model = copy.deepcopy(self.model), name=name_lists[i], test_data = None)
            edge_name = name_lists[i]
            self.edges.append(edge)
            self.edges_names.append(edge_name)
            self.edge_to_name_mapping[edge] = edge_name
            self.name_to_edge_mapping[edge_name] = edge

    def initialize_clients(self):
        for client in self.clients:
            self.client_to_name_mapping[client] = client.name
            self.name_to_client_mapping[client.name] = client


    def initialize_client_to_server(self):
        self.client_edge_mapping = {}
        self.edge_client_mapping = {}
        for edge in self.edges:
            # print(edge.name)
            self.edge_client_mapping[edge.name] = []
            
        for client in self.clients:
            client.current_edge_name = np.random.choice(self.edges_names,1)[0]
            self.client_edge_mapping[client.name] = client.current_edge_name
            self.edge_client_mapping[client.current_edge_name].append(client.name)

    def print_clients_info(self):
        print("Current number of clients: ", self.current_num_clients)
        for client in self.clients:
            client.print_client_info()

    def initialize(self):
        self.initialize_edges()
        self.initialize_clients()
        self.initialize_client_to_server()
        # print(self.client_edge_mapping)

    # def sample_data_with_replacement(self, num_clients):
    #     client_data_lists = []
    #     training_size = self.x_train.shape[0]
    #     for i in range(num_clients):
    #         chosen_indices = random.sample([idx for idx in range(training_size)], self.num_data_samples_per_client)
    #         client_X = self.x_train[chosen_indices]
    #         client_Y = self.y_train[chosen_indices]

    #         client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(client_X, client_Y, 
    #                                                                                           test_size = self.option['client_valid_ratio'],
    #                                                                                           random_state=self.option['seed'])
    #         # print(client_X_train)
    #         client_train_dataset = XYDataset(client_X_train, client_Y_train)
    #         client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
    #         # print(client_X_train.shape, client_X_valid.shape, client_Y_train.shape, client_Y_valid.shape)

    #         client_data_lists.append( (client_train_dataset, client_valid_dataset) )
        
    #     return client_data_lists


    # def sample_data_without_replacement(self, num_clients):
    #     client_data_lists = []
    #     training_size = self.x_train.shape[0]
    #     # print("X train", self.x_train.shape)
    #     if self.option['non_iid_classes'] == 0:
    #         client_indices_split = np.split(np.array([idx for idx in range(training_size)]), num_clients)
    #         for i in range(num_clients):
    #             chosen_indices = client_indices_split[i]
    #             client_X = self.x_train[chosen_indices]
    #             client_Y = self.y_train[chosen_indices]
    #             # print(client_X.shape,client_Y.shape)

    #             client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(client_X, client_Y, 
    #                                                                                             test_size = self.option['client_valid_ratio'])
    #             # print("Client X train",client_X_train)

    #             client_train_dataset = XYDataset(client_X_train, client_Y_train)
    #             client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
    #             # print(client_X_train.shape, client_X_valid.shape, client_Y_train.shape, client_Y_valid.shape)

    #             client_data_lists.append( (client_train_dataset, client_valid_dataset) )
        
    #     else:
    #         non_iid_data_lists = []
    #         all_classes = list(np.unique(self.y_train))
    #         num_classes = len(all_classes)
    #         # print(all_classes, num_classes)
    #         num_partitions_per_class = num_clients // num_classes
    #         partition_size = self.x_train.shape[0] // num_clients
    #         print("Number of partitions per class", num_partitions_per_class)
    #         print("partition size", partition_size)
    #         # print(self.y_train.shape)
    #         for label in all_classes:
    #             label_indices = np.argwhere(self.y_train == label)
    #             x_train_label = self.x_train[label_indices].squeeze(0)
    #             y_train_label = self.y_train[label_indices]
    #             y_train_label = y_train_label.squeeze(0)
    #             print(x_train_label.shape, y_train_label.shape)
    #             for i in range(num_partitions_per_class):
    #                 x_train_label_partition = x_train_label[partition_size * i: partition_size * (i+1)]
    #                 y_train_label_partition = y_train_label[partition_size * i: partition_size * (i+1)]
    #                 print("partition shape: ", x_train_label_partition.shape, y_train_label_partition.shape)
    #                 client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(x_train_label_partition, y_train_label_partition, 
    #                                                                                                 test_size = self.option['client_valid_ratio'])

    #                 # print(client_X_train.shape, client_Y_train.shape, client_X_valid.shape)
    #                 client_train_dataset = XYDataset(client_X_train, client_Y_train)
    #                 client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
    #                 non_iid_data_lists.append( (client_train_dataset, client_valid_dataset) )
    #         client_data_lists = non_iid_data_lists

    #                 # print(y_train_label_partition, x_train_label_partition, x_train_label_partition.shape)
    #             # print(x_train_label[0:10], y_train_label[0:10]
    #         # print(np.unique(self.y_train))

        
    #     return client_data_lists
        
        # print(self.client_edge_mapping)




class EdgeServer(BasicEdgeServer):
    def __init__(self, option,model, name = '', clients = [], test_data=None):
        super(EdgeServer, self).__init__(option,model, name , clients , test_data)
        self.clients = []

        self.model_buffer = []
        self.max_buffer_size = option['edge_model_buffer_size'] 
        self.ensemble_model = None

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
        if self.max_buffer_size == 3:
            # weight = [0.1,0.2,0.7]
            weight = [0.33,0.33,0.33]

        if self.max_buffer_size == 6:
            weight = [0.025,0.025,0.05,0.1,0.2,0.6]
            weight = [1/6,1/6,1/6,1/6,1/6,1/6]

        # if self.max_buffer_size == 6:
        #     weight = [1/30,1/30,1/30,1/30,1/30,1/30,0.1,0.6]
        else:
            weight = [1/self.max_buffer_size for _ in range(self.max_buffer_size)]

        self.ensemble_model = fmodule._model_average(self.model_buffer,weight)
    

    def print_edge_info(self):
        print('Edge {} - cover area: {}'.format(self.name,self.cover_area))

    def communicate(self, clients,global_model):
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
                response_from_edge = self.communicate_with(client,global_model)
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

    def communicate_with(self, client,  global_model):
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
            "model" : copy.deepcopy(self.model),
            "edge_ensemble_model" : copy.deepcopy(self.ensemble_model),
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

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        valid_losses = [cp["valid_loss"] for cp in packages_received_from_clients]
        train_acc = [cp["train_acc"] for cp in packages_received_from_clients]
        valid_acc = [cp["valid_acc"] for cp in packages_received_from_clients]

        return models, (train_losses, valid_losses, train_acc, valid_acc)



class MobileClient(BasicMobileClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(option,  name, train_data, valid_data)
        # self.velocity = velocity
        self.alpha = option['distill_alpha']
        self.global_beta = option['global_beta']
        self.T  = option['distill_temperature']

        if 'cifar100' in option['task']:
            num_classes = 100
        else:
            num_classes = 10

        self.option = option
        self.alpha = 0.1
        self.beta = 0.8
        self.T = 0.2
        self.distill_loss = SoftTargetDistillLoss( T = self.T)
        # self.distill_loss = torch.nn.MSELoss()

        self.associated_server = None

    
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
        return received_pkg['model'], received_pkg['edge_ensemble_model'], received_pkg['global_model']

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
        model, edge_ensemble_model, global_model = self.unpack(svr_pkg)
        # print("CLient unpacked to package")
        train_loss = self.train_loss(model)
        valid_loss = self.valid_loss(model)
        train_acc = self.train_metrics(model)
        valid_acc = self.valid_metrics(model)

        # print("Client evaluated the train losss")
        self.train(model,edge_ensemble_model, global_model)
        # print("Client trained the model")
        eval_dict = {'train_loss': train_loss, 
                      'valid_loss': valid_loss,
                      'train_acc':train_acc,
                      'valid_acc': valid_acc}
        cpkg = self.pack(model,global_model, eval_dict)
        # print("Client packed and finished")
        return cpkg

    def pack(self, model,global_model, eval_dict ):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        pkg = {'model': model, 'global_model':global_model} | eval_dict
        return pkg


    def train(self, edge_model, edge_ensemble_model, global_model):
        edge_teacher = copy.deepcopy(edge_ensemble_model)
        global_teacher = copy.deepcopy(global_model)
        if edge_teacher != None:
            edge_teacher.freeze_grad()
        if global_teacher != None:
            global_teacher.freeze_grad()

        edge_model.train()

        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, edge_model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                optimizer.zero_grad()

                if edge_teacher != None and global_teacher != None:
                    tdata = self.calculator.data_to_device(batch_data)
                    input, target = tdata[0], tdata[1].type(torch.LongTensor)
                    target = target.to(input.device)
                    output_local_model = edge_model(input)

                    with torch.no_grad():
                        output_edge_model = edge_teacher(input)
                        output_global_model = global_teacher(input)

                    distill_loss = ( (1-self.global_beta) * self.distill_loss(output_local_model, output_edge_model) \
                                    +self.global_beta * self.distill_loss(output_local_model, output_global_model)) 
                    original_loss = self.calculator.get_loss(edge_model, batch_data)
                    loss = self.alpha * distill_loss +  original_loss
                
                else:
                    loss = self.calculator.get_loss(edge_model, batch_data)

                loss.backward()
                optimizer.step()

        self.model = copy.deepcopy(edge_model)

        return
                