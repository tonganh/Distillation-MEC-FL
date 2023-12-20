import random
from utils import fmodule
import sys
sys.path.append('..')
from  algorithm.fedbase import BasicServer, BasicClient
import copy
import math
import numpy as np
from main_mobile import logger


class BasicCloudServer(BasicServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(BasicCloudServer, self).__init__(option, model, clients, test_data)
        # self.clients = []
        # #print(clients)
        
        self.mean_num_clients = option['num_clients']
        self.current_num_clients = self.mean_num_clients

        self.num_edges = option['num_edges']
        self.edges = []
        self.edges_names = []

        self.p_move = option['p_move']

        self.edge_update_frequency = option['edge_update_frequency']

        # self.sample_with_replacement = option['sample_with_replacement']
        self.client_edge_mapping = {}
        self.edge_client_mapping = {}

        self.name_to_edge_mapping = {}
        self.edge_to_name_mapping = {}
        self.client_to_name_mapping = {}
        self.name_to_client_mapping = {}


        self.option = option

        # List to store clients currently in range
        self.selected_clients = []
        # List to store clients currently out of range
        self.client_train_losses = []
        self.client_valid_losses = []
        self.client_train_metrics = []
        self.client_valid_metrics = []
        self.client_to_edge_communication_cost = []
        self.edge_to_cloud_communication_cost = []
        self.deleted_clients_name = []


    def get_model_size(self, model):
        num_params = sum(p.numel() for p in model.parameters())
        # Mỗi tham số là một số thực dấu chấm động 32 bit, tức là 4 byte
        num_bytes = num_params * 4
        return num_bytes



    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
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


        all_client_train_losses = []
        all_client_valid_losses = []
        all_client_train_metrics = []
        all_client_valid_metrics = []
        communication_cost_in_round = 0
        trained_clients = []

        for edge in self.edges:

            np.random.seed(self.option['seed'] +21)

            clients_chosen_in_edge =     list(np.random.choice(self.edge_client_mapping[edge.name],
                                                               int(len(self.edge_client_mapping[edge.name]) * self.option['proportion']), replace=False))
            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in clients_chosen_in_edge:
                    aggregated_clients.append(client)
                    trained_clients.append(client.name)

                    if client.name == 'Client010':
                        print("Edge that sent client 10 the model: ", edge.name)

            if len(aggregated_clients) > 0:
                # print(aggregated_clients)
                # print(edge.communicate(aggregated_clients))
                aggregated_clients_models , (agg_clients_train_losses, 
                                             agg_clients_valid_losses, 
                                             agg_clients_train_accs, 
                                             agg_clients_valid_accs)= self.communicate(aggregated_clients)
                total_size_edge_1side_cloud = sum(self.get_model_size(model) for model in aggregated_clients_models)
                # !nhân 2 vì có cả chiều đi và chiều về
                communication_cost_in_round += total_size_edge_1side_cloud*2

                # edge_total_datavol = sum([client.datavol for client in aggregated_clients])
                # edge.total_datavol = edge_total_datavol
                # aggregation_weights = [client.datavol / edge_total_datavol for client in aggregated_clients]
                # edge.model =  self.aggregate(aggregated_clients_models, p = aggregation_weights)

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




        for edge in self.edges:

            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in self.edge_client_mapping[edge.name]:
                    if client.name in trained_clients:
                        # print(edge.name,client.name)
                        aggregated_clients.append(client)
                        if client.name == 'Client010':
                            print('aggregating edge for client 10: ', edge.name)
    
            if len(aggregated_clients) > 0:
                # print('Chosen')
                aggregated_clients_models = [client.model for client in aggregated_clients]
                edge_total_datavol = sum([client.datavol for client in aggregated_clients])
                edge.total_datavol = edge_total_datavol
                aggregation_weights = [client.datavol / edge_total_datavol for client in aggregated_clients]
                edge.model =  self.aggregate(aggregated_clients_models, p = aggregation_weights)


        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        # models = [edge.model for edge in self.edges]

        communication_cost_edge_cloud = 0

        if t % self.edge_update_frequency == 0:
            models = [edge.model for edge in self.edges]

            # communication_cost = len(model) * 2 * size(models[0])  edg2cloud = communication_cost
            models = [edge.model for edge in self.edges]
            total_size_edge_1side_cloud = sum(self.get_model_size(model) for model in models)
            communication_cost_edge_cloud = total_size_edge_1side_cloud*2
            sum_datavol = sum([edge.total_datavol for edge in self.edges])
            edge_weights = [edge.total_datavol / sum_datavol for edge in self.edges]
            self.model = self.aggregate(models, p = edge_weights)

            for edge in self.edges:
                edge.model = copy.deepcopy(self.model)

        self.edge_to_cloud_communication_cost.append(communication_cost_edge_cloud)
        self.client_to_edge_communication_cost.append(communication_cost_in_round)

        edges_models_list = []
        for edge in self.edges:
                edges_models_list.append(copy.deepcopy(edge.model))
     
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
        
    
    def update_client_server_mapping(self):
        # print(self.edges_names)
        # self.client_edge_mapping = {}
        self.edge_client_mapping = {}
        for edge in self.edges:
            self.edge_client_mapping[edge.name] = []

        for client in self.clients:
            current_edge_name = self.client_edge_mapping[client.name]
            edge_index = self.edges_names.index(current_edge_name)
            # print("Old name" , current_edge_name)
            if 0 < edge_index < self.num_edges -1 :
                client_edge_moving_choice = self.edges_names[edge_index -1 : edge_index + 2]
                # print("Case 1: ", client_edge_moving_choice)

            elif edge_index == 0:
                # print(self.edges_names[:2])
                # print(self.edges_names[-1])
                client_edge_moving_choice = [self.edges_names[-1]]  + self.edges_names[:2] 
                # print("Case 2: ", client_edge_moving_choice)
            else:
                client_edge_moving_choice =  self.edges_names[-2:]  + [self.edges_names[0]] 
                # print("Case 3: ", client_edge_moving_choice)

                
            p = [self.p_move/2,1-self.p_move,self.p_move/2]

            if client.name == 'Client010':
                print(client_edge_moving_choice)
                print(p)

            new_edge_name = np.random.choice(client_edge_moving_choice, 1, p=p,replace = False)[0]
            # print("New name: ", new_edge_name)
            client.current_edge_name = new_edge_name
            self.client_edge_mapping[client.name] = client.current_edge_name
            self.edge_client_mapping[client.current_edge_name].append(client.name)
        

    def get_clients_names(self):
        clients_name = [client.name for client in self.clients]
        return clients_name
      
    def delete_clients(self, count_remove_clients):
        clients_after_delete = random.sample(self.clients, k=len(self.clients)-count_remove_clients)
        clients_name_initial = [client.name for client in self.clients]
        clients_name_deleted = [client.name for client in clients_after_delete]
        
        # Tạo tập hợp các tên khách hàng duy nhất
        clients_name_initial_set = set(clients_name_initial)
        clients_name_deleted_set = set(clients_name_deleted)
        
        # Lấy danh sách các tên khách hàng đã bị xóa
        removed_clients_names = list(clients_name_initial_set - clients_name_deleted_set)
        
        print(f'Before delete, total clients is: {len(self.clients)}')
        self.clients = clients_after_delete
        print(f'After delete, total clients is: {len(self.clients)}')
        self.deleted_clients_name.extend(removed_clients_names)
        print(f'Name deleted client: {self.deleted_clients_name}')
        return removed_clients_names

    # def global_update_location(self):
    #     new_client_list = []
    #     for client in self.clients:
    #         # client.print_client_info()
    #         # print(client.location)
    #         client.update_location()
    #         # print(client.location)
    #         new_client_list.append(client)
    #     self.clients = new_client_list

    # def assign_client_to_server(self):
    #     client_buffer = self.clients.copy()
    #     for edge in self.edges:
    #         edge_area = edge.cover_area
    #         edge_name = edge.name
    #         if edge not in self.client_edge_mapping:
    #             self.client_edge_mapping[edge_name] = []
    #         for client in client_buffer:
    #             if edge_area[0] <= client.location <= edge_area[1]:
    #                 self.client_edge_mapping[edge_name].append(client.name)
    #     # print(self.client_edge_mapping)
    #                 # print(self.client_edge_mapping)
        
    # def update_client_list(self):
    #     filtered_client_list = []
    #     filtered = 0
    #     for client in self.clients:
    #         # client.print_client_info()
    #         if self.left_road_limit <= client.location <= self.right_road_limit:
    #             filtered_client_list.append(client)
    #             # print(True)
    #         else:
    #             self.unused_clients_queue.append(client)
    #             filtered +=1
    #             # print(False)
    #     # print("Number of filtered clients",filtered)
    #     self.clients = filtered_client_list
    #     clients_names  = self.get_clients_names()
    #     if len(self.clients) < self.mean_num_clients - self.std_num_clients:
    #         self.current_num_clients = np.random.randint(low = self.mean_num_clients - self.std_num_clients, high=self.mean_num_clients+self.std_num_clients + 1,
    #                                                     size=1)[0]
    #         num_clients_to_readd = self.current_num_clients - len(self.clients)
    #         if num_clients_to_readd < len(self.unused_clients_queue):
    #             clients_to_readd = random.sample(self.unused_clients_queue, k = num_clients_to_readd)
    #             for client in clients_to_readd:
    #                 client.location =np.random.randint( self.left_road_limit, self.right_road_limit, size =1)[0]
    #                 client_name = client.name
    #                 if client_name not in clients_names and client_name not in self.deleted_clients_name:
    #                     self.clients.append(client)
    #                     clients_names.append(client_name)
    #                     print(f'Line 175 - {len(self.clients)}')
    #                 # client.location = client
    #         else:
    #             clients_to_readd = self.unused_clients_queue
    #             for client in clients_to_readd:
    #                 client.location =np.random.randint( self.left_road_limit, self.right_road_limit, size =1)[0]
    #                 client_name = client.name
    #                 if client_name not in clients_names and client_name not in self.deleted_clients_name:
    #                     self.clients.append(client)
    #                     clients_names.append(client_name)
    #                     print(f'Line 208 - {len(self.clients)}')
                        
    #     if len(self.clients) > 100:
    #         import pdb; pdb.set_trace()

    
    # def initialize_clients_location_velocity(self):
    #     new_client_list = []

    #     np.random.seed(self.option['seed'] +21)

    #     locations = np.random.randint( self.left_road_limit, self.right_road_limit, size = len(self.clients))
    #     if self.option['mean_velocity'] != 0:
    #         velocities_absolute = np.random.randint( self.mean_velocity - self.std_velocity, self.mean_velocity + self.std_velocity, size = len(self.clients))
    #         np.random.seed(self.option['seed'] +21)
    #         velocities_direction = np.array([random.choice([-1,1]) for i in range(len(self.clients))])
    #         velocities = velocities_absolute * velocities_direction
    #         # print(velocities_direction, velocities)
    #     else:
    #         velocities = np.array([0 for i in range(len(self.clients))])

    #     for i  in range(len(self.clients)):
    #         client = self.clients[i]
    #         client.location = locations[i] 
    #         client.velocity = velocities[i]
    #         new_client_list.append(client)
        
    #     self.clients = new_client_list
        


    def print_edges_info(self):
        print("Current number of edges: ", self.num_edges)
        for edge in self.edges:
            edge.print_edge_info()
    
    # def print_client_info(self):
    #     for client in self.clients
            

    def initialize(self):
        self.initialize_edges()
        self.initialize_clients()
        self.initialize_client_to_server()
        # self.assign_client_to_server()


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

    def aggregate(self, models, p=[]):
        """
        Aggregate the locally improved models.
        :param
            models: a list of local models
            p: a list of weights for aggregating
        :return
            the averaged result

        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        -------------------------------------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)   |other
        ==============================================================================================|============================
        N/K * Σpk * model_k                 |1/K * Σmodel_k                  |(1-Σpk) * w_old + Σpk * model_k     |Σ(pk/Σpk) * model_k
        """
        # print('aggregating')
        # print(self.agg_option)
        if not models: 
            return self.model
        if self.agg_option == 'model_sim':
            print('model_sim')
            # for model in models:
            #     print(fmodule._model_cossim(self.model,model))
            if self.model == None:
                cos_sim_norm = [1 for _ in range(len(models))]
            else:
                for model in models:
                    print(fmodule._modeldict_cossim(self.model.state_dict(),model.state_dict()))

                cos_sim = np.array([fmodule._model_cossim(self.model,model).detach().cpu().numpy() for model in models])
                print(cos_sim)
                cos_sim_norm = (cos_sim-np.min(cos_sim))/(np.max(cos_sim)-np.min(cos_sim))
                cos_sim_norm = [sim_score/np.sum(cos_sim_norm) for sim_score in cos_sim_norm]
                print(cos_sim_norm)
            p = cos_sim_norm * p
            #print(cos_sim_norm)
            return fmodule._model_average(models, p=p)

        elif self.agg_option == 'weighted_scale':
            K = len(models)
            N = self.num_clients
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]) * N / K
        elif self.agg_option == 'uniform':
            return fmodule._model_average(models, p=p)
        elif self.agg_option == 'weighted_com':
            w = fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])
            return (1.0-sum(p))*self.model + w
        else:
            sump = sum(p)
            p = [pk/sump for pk in p]
            return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)])



class BasicEdgeServer(BasicServer):
    def __init__(self, option,model, name = '', clients = [], test_data=None):
        super(BasicEdgeServer, self).__init__(option, model, clients, test_data)
        self.name = name
        self.option = option
        self.total_datavol = 0
        # self.list_clients = []
        self.agg_option = self.option['aggregate']
        # self.model = None

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



    # def update_list_clients(self,clients):
    #     self.list_clients = clients

    

class BasicMobileClient(BasicClient):
    def __init__(self, option,  name='', train_data=None, valid_data=None):
        super(BasicMobileClient, self).__init__(option, name, train_data, valid_data)
        self.current_edge_name = None
        # self.mean_velocity = mean_velocity
        # self.std_velocity = std_velocity
        # self.current_velocity = mean_velocity
    
    # def get_current_velocity(self):
    #     self.current_velocity = np.random.randint(low=self.mean_velocity - self.std_velocity, high=self.mean_velocity + self.std_velocity, size = 1)[0]

    def get_current_edge(self):
        return self.current_edge_name


    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                # print(batch_data)
                # print(batch_data.shape)
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                optimizer.step()
        self.model = copy.deepcopy(model)
        return

    def test(self, model, dataflag='valid'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):

            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss

    def unpack(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model']

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
        # #print("In reply function of client")
        model = self.unpack(svr_pkg)

        # #print("CLient unpacked to package")
        # #print("Client evaluated the train losss")
        self.train(model)

        train_loss = self.train_loss(model)
        valid_loss = self.valid_loss(model)
        train_acc = self.train_metrics(model)
        valid_acc = self.valid_metrics(model)



        # #print("Client trained the model")
        eval_dict = {'train_loss': train_loss, 
                      'valid_loss': valid_loss,
                      'train_acc':train_acc,
                      'valid_acc': valid_acc}
        cpkg = self.pack(model, eval_dict)
        # #print("Client packed and finished")
        return cpkg

    def pack(self, model,eval_dict ):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        pkg = {'model': model} | eval_dict
        return pkg


    def train_loss(self, model):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')[1]

    def train_metrics(self, model):
        """
        Get the task specified metrics of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')[0]

    def valid_loss(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)[1]

    def valid_metrics(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)[0]


