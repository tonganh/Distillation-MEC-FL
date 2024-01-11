from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader, WholeTaskReader


class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=10, skewness=0.5):
        super(TaskGen, self).__init__(benchmark='svhn',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/svhn/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
            (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))])
        self.train_data = datasets.SVHN(
            self.rawdata_path, split='train', download=True, transform=transform)

        self.test_data = datasets.SVHN(
            self.rawdata_path, split='test', download=True, transform=transform)

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist()
                   for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1]
                   for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist()
                  for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x': train_x, 'y': train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return


class TaskReader(XYTaskReader):
    def __init__(self, taskpath=''):
        super(TaskReader, self).__init__(taskpath)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
