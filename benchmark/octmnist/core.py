from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader, WholeTaskReader
from medmnist import OCTMNIST
import numpy as np


class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=10, skewness=0.5, number_class_per_client=2):
        super(TaskGen, self).__init__(benchmark='octmnist',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/octmnist/data',
                                      number_class_per_client=number_class_per_client
                                      )
        self.num_classes = 4
        self.save_data = self.XYData_to_json

    def load_data(self):
        self.train_data = OCTMNIST(split="train", download=True, root=self.rawdata_path, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = OCTMNIST(split="test", download=True, root=self.rawdata_path, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        new_train_data = []
        for i in range(len(self.train_data)):
            new_train_data.append(
                (self.train_data[i][0], int(self.train_data[i][1][0])))
        self.train_data = new_train_data

        new_test_data = []
        for i in range(len(self.test_data)):
            new_test_data.append(
                (self.test_data[i][0], int(self.test_data[i][1][0])))
        self.test_data = new_test_data

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist()
                   for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1]
                   for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist()
                  for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1]
                  for did in range(len(self.test_data))]
        self.train_data = {'x': train_x, 'y': train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return


class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
