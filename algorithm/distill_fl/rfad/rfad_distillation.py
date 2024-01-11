import numpy as np
import matplotlib.pyplot as plt
from .data import get_zca_matrix, transform_data, get_cifar10
from .utils_rfad import get_random_features, double_print, mapping_data_with_index
from .utils_rfad import *
import torch_optimizer as torch_optim
import random
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from functools import partial
import time
import os
from .models import ConvNet_wide
from .coresets import make_coreset


class RFAD_Distillation():
    def __init__(self, ipc=1, save_path='results_rfad', server_gpu_id='0', dataset='cifar10', n_iters=5000):
        self.ipc = ipc
        self.save_path = save_path
        self.server_gpu_id = 'cuda:{}'.format(
            server_gpu_id) if torch.cuda.is_available() and server_gpu_id != -1 else 'cpu'
        self.dataset = dataset
        self.n_iters = n_iters
        self.reversed_data_mapping = {}
        pass

    def corrupt_data(self, X, corruption_mask, X_init, whitening_mat):
        # return X
        X_corrupted = ((1 - corruption_mask) * X) + (corruption_mask * X_init)

        if not whitening_mat is None:
            print('slls')
            X_corrupted = transform_data(X_corrupted, whitening_mat)
        return X_corrupted

    def distill_dataset(self, X_train, y_train, model_class, lr, n_models, n_batches,
                        iters=10000, platt=False, ga_steps=1, schedule=None, save_location=None,
                        samples_per_class=1, n_classes=10, learn_labels=False, batch_size=1280,
                        X_valid=None, y_valid=None, n_channels=3, im_size=32, X_init=None, jit=1e-6,
                        seed=0, corruption=0, whitening_mat=None, from_loader=False):

        y_train = y_train.long()
        coreset_size = samples_per_class * n_classes

        X_coreset = torch.nn.Parameter(torch.empty(
            (coreset_size, n_channels, im_size, im_size), device=self.server_gpu_id).normal_(0, 1))
        transform_mat = torch.nn.Parameter(torch.empty(
            (n_channels * im_size * im_size, n_channels * im_size * im_size), device=self.server_gpu_id).normal_(0, 1))
        transform_mat.data = torch.eye(
            n_channels * im_size * im_size, device=self.server_gpu_id)

        # parameter used for platt scaling
        k = torch.nn.Parameter(torch.tensor(
            (0.), device=self.server_gpu_id).double())

        y_coreset = torch.nn.Parameter(torch.empty(
            (coreset_size, n_classes), device=self.server_gpu_id).normal_(0, 1))
        print("Line 59 - onehot")
        y_coreset.data = (torch.Tensor(one_hot(np.concatenate([[j for i in range(
            samples_per_class)] for j in range(n_classes)]), n_classes, check_debug=False)).float().cuda(device=self.server_gpu_id) - 1/n_classes)

        if not X_init is None:
            X_coreset.data = X_init.cuda(device=self.server_gpu_id)
        else:
            X_init = X_coreset.data.clone()
        X_init = X_init.cuda(device=self.server_gpu_id).clone()

        if not whitening_mat is None:
            whitening_mat = whitening_mat.cuda(device=self.server_gpu_id)

        if corruption > 0:
            torch.manual_seed(seed)
            corruption_mask = (torch.rand(size=X_coreset.shape) < corruption).int(
            ).float().cuda(device=self.server_gpu_id)  # 0 = don't corrupt, 1 = corrupt

        losses = []

        if platt:
            if not learn_labels:
                optim = torch_optim.AdaBelief([{"params": [X_coreset]},
                                               {"params": [transform_mat], "lr": 5e-5}, {"params": [k], "lr": 1e-2}], lr=lr, eps=1e-16)  # a larger learning rate for k usually helps
            else:
                optim = torch_optim.AdaBelief([{"params": [X_coreset, y_coreset]},
                                               {"params": [transform_mat], "lr": 5e-5}, {"params": [k], "lr": 1e-2}], lr=lr, eps=1e-16)
        else:
            if not learn_labels:
                optim = torch_optim.AdaBelief([{"params": [X_coreset]},
                                               {"params": [transform_mat], "lr": 5e-5}], lr=lr, eps=1e-16)
            else:
                optim = torch_optim.AdaBelief([{"params": [X_coreset, y_coreset]},
                                               {"params": [transform_mat], "lr": 5e-5}], lr=lr, eps=1e-16)

        model_rot = 10
        schedule_i = 0

        valid_fixed_seed = (np.abs(seed) + 1) * np.array(list(range(16)))

        if X_valid is not None:
            X_valid_features, _ = get_random_features(
                X_valid, model_class, 16, 4096, fixed_seed=valid_fixed_seed, device=self.server_gpu_id)
            y_valid_one_hot = one_hot(
                y_valid, n_classes, check_debug=False) - 1/n_classes
        X_coreset_best = None
        y_coreset_best = None
        k_best = None

        best_iter = -1
        best_valid_loss = np.inf
        acc = 0

        start_time = time.time()
        output_file = None

        if save_location is not None:
            if not os.path.isdir(save_location):
                os.makedirs(save_location)
            output_file = open(
                '{}/training_log.txt'.format('./'), 'a')

        file_print = partial(double_print, output_file=output_file)

        if from_loader:
            X_iterator = iter(X_train)

        for i in range(iters):
            if i % (ga_steps * 40) == 0:
                file_print(acc)
                transformed_coreset = transform_data(
                    X_coreset.data, transform_mat.data)

                if corruption > 0:
                    transformed_coreset = self.corrupt_data(
                        transformed_coreset, corruption_mask, X_init, whitening_mat)

                # if save_location is not None:
                #     np.savez('{}/{}.npz'.format(save_location, i), images=transformed_coreset.data.cpu(
                #     ).numpy(), labels=y_coreset.data.cpu().numpy(), k=k.data.cpu(), jit=jit)

                # get validation acc

                X_coreset_features, _ = get_random_features(
                    transformed_coreset.cpu(), model_class, 16, 4096, fixed_seed=valid_fixed_seed, device=self.server_gpu_id)
                K_xx = 2 * (X_coreset_features @ X_coreset_features.T) + 0.01
                K_xx = K_xx + (jit * np.eye(1 * coreset_size)
                               * np.trace(K_xx)/coreset_size)
                solved = np.linalg.solve(K_xx.astype(
                    np.double), y_coreset.data.cpu().numpy().astype(np.double))
                preds_valid = (
                    2 * (X_valid_features @ X_coreset_features.T) + 0.01).astype(np.double) @ solved

                if not platt:
                    valid_loss = 0.5 * \
                        np.mean((y_valid_one_hot - preds_valid)**2)

                else:
                    valid_loss = nn.CrossEntropyLoss()(torch.exp(k) * torch.tensor(preds_valid).cuda(device=self.server_gpu_id),
                                                       y_valid.long().cuda(device=self.server_gpu_id)).detach().cpu().item()
                    print("Pass - 158")
                # 0.001
                valid_acc = np.mean(preds_valid.argmax(
                    axis=1) == y_valid_one_hot.argmax(axis=1))
                total_time_running_for_iter = time.time() - start_time
                file_print('iter: {}, valid loss: {}, valid acc: {}, elapsed time: {:.1f}s'.format(
                    i, valid_loss, valid_acc, time.time() - start_time))

                if valid_loss < best_valid_loss:
                    X_coreset_best = X_coreset.data.detach().clone()
                    transform_mat_best = transform_mat.data.detach().clone()
                    y_coreset_best = y_coreset.data.detach().clone()
                    k_best = k.data.detach().clone()
                    best_iter = i
                    best_valid_loss = valid_loss

                patience = 1000
                if (i > best_iter + (ga_steps * patience) and i > schedule[-1][0] + (ga_steps * patience)) or iters == 1 or valid_loss < 0.001 or total_time_running_for_iter > 120:
                    file_print('early stopping at iter {}, reverting back to model from iter {}'.format(
                        i, best_iter))
                    transformed_best_coreset = transform_data(
                        X_coreset_best.data, transform_mat_best.data)

                    if corruption > 0:
                        transformed_best_coreset = self.corrupt_data(
                            transformed_best_coreset, corruption_mask, X_init, whitening_mat)

                    print("self.save_path: ", self.save_path)
                    torch.save(transformed_best_coreset.data,
                               f'{self.save_path}/x_distill.pt')
                    predicted_label_best = torch.argmax(
                        y_coreset_best.data, dim=1)
                    predicted_label_best_list = predicted_label_best.tolist()
                    real_values = list(
                        map(lambda x: self.reversed_data_mapping[x], predicted_label_best_list))
                    real_values_tensor = torch.tensor(real_values).cpu()
                    torch.save(np.array(real_values_tensor),
                               f'{self.save_path}/y_distill.pt')

                    # np.savez('{}/best.npz'.format(save_location), images=transformed_best_coreset.data.cpu().numpy(),
                    #          labels=y_coreset_best.data.cpu().numpy(), valid_loss=best_valid_loss, k=k_best.data.cpu().numpy(), jit=jit, best_iter=best_iter)
                    return transformed_best_coreset, y_coreset_best.data

            if schedule is not None and schedule_i < len(schedule):
                if i >= schedule[schedule_i][0]:
                    file_print("UPDATING MODEL COUNT: {}".format(
                        schedule[schedule_i]))
                    n_models = schedule[schedule_i][1]
                    model_rot = schedule[schedule_i][2]
                    schedule_i += 1

            if i % ga_steps == 0:
                optim.zero_grad()

            if i % model_rot == 0:
                if i != 0:
                    del models_list

                models_list = []
                rand_seed = random.randint(0, 50000)
                torch.manual_seed(rand_seed)
                for m in range(n_models):
                    models_list.append(model_class(
                        n_random_features=4096, chopped_head=True))

                    models_list[-1].to(self.server_gpu_id)
                    models_list[-1].eval()

            X_coreset_features = []
            transformed_data = transform_data(X_coreset, transform_mat)
            if corruption > 0:
                transformed_data = self.corrupt_data(
                    transformed_data, corruption_mask, X_init, whitening_mat)

            for m in range(n_models):
                X_coreset_features.append(models_list[m](transformed_data))
            X_coreset_features = torch.cat(
                X_coreset_features, 1)/np.sqrt(n_models * X_coreset_features[0].shape[1])

            K_xx = (2 * X_coreset_features @ X_coreset_features.T) + 0.01
            K_xx = K_xx + (jit * torch.eye(1 * coreset_size,
                                           device=self.server_gpu_id) * torch.trace(K_xx)/coreset_size)

            X_train_features = []
            y_values = []
            with torch.no_grad():
                for b in range(n_batches):
                    if not from_loader:
                        indices = np.random.choice(
                            X_train.shape[0], 10, replace=False)
                        X_batch = X_train[indices].float().cuda(
                            device=self.server_gpu_id)
                        y_batch = y_train[indices]
                    else:
                        try:
                            batch = next(X_iterator)
                        except StopIteration:
                            X_iterator = iter(X_train)
                            batch = next(X_iterator)
                        X_batch = batch[0].cuda(device=self.server_gpu_id)
                        y_batch = batch[1]

                    X_train_features_inner = []

                    for m in range(n_models):
                        X_train_features_inner.append(
                            models_list[m](X_batch).detach())

                    y_values.append(torch.nn.functional.one_hot(y_batch, n_classes).cuda(
                        device=self.server_gpu_id) - 1/n_classes)

                    X_train_features_inner = torch.cat(
                        X_train_features_inner, 1)/np.sqrt(n_models * X_train_features_inner[0].shape[1])

                    X_train_features.append(X_train_features_inner)

                X_train_features = torch.cat(X_train_features, 0).detach()
                y_values = torch.cat(y_values, 0)

            solved = torch.linalg.solve(K_xx.double(), y_coreset.double())
            K_zx = 2 * (X_train_features @ X_coreset_features.T) + 0.01
            preds = K_zx.double() @ solved

            acc = np.mean(preds.detach().cpu().numpy().argmax(axis=1)
                          == y_values.cpu().numpy().argmax(axis=1))

            if platt:
                loss = nn.CrossEntropyLoss()(torch.exp(k) * preds, torch.argmax(y_values, 1))
            else:
                loss = .5 * torch.mean((y_values - preds)**2)

            if i % ga_steps == (ga_steps - 1):
                loss.backward()

            losses.append((loss).detach().cpu().numpy().item())

            if i % ga_steps == (ga_steps - 1):

                optim.step()
                file_print('=', end='')

    def distill(self, X_TRAIN_RAW, LABELS_TRAIN, X_TEST_RAW, LABELS_TEST, options):
        # X_TRAIN_RAW = torch.cat((X_TRAIN_RAW, X_TEST_RAW), 0)
        # LABELS_TRAIN = torch.cat((LABELS_TRAIN, LABELS_TEST), 0)
        LABELS_TRAIN, reversed_data_mapping = mapping_data_with_index(
            LABELS_TRAIN)
        LABELS_TEST, _ = mapping_data_with_index(LABELS_TEST)
        self.reversed_data_mapping = reversed_data_mapping
        if self.dataset == 'cifar10':
            im_size = 32
            n_channels = 3
        if self.dataset == 'mnist':
            im_size = 28
            n_channels = 1
        if self.dataset == 'fashion_mnist':
            im_size = 28
            n_channels = 1
        if self.dataset == 'cifar100':
            im_size = 32
            n_channels = 3
        if self.dataset == 'svhn':
            im_size = 32
            n_channels = 3

        classes = np.unique(LABELS_TRAIN)
        n_classes = len(classes)

        if self.dataset != 'celeba':
            np.random.seed(options['seed'])
            valid_indices = []
            for c in classes:
                class_indices = np.where(LABELS_TRAIN == c)[0]
                valid_indices.append(class_indices[np.random.choice(
                    len(class_indices), 500 if n_classes == 10 else 100)])

            valid_indices = np.concatenate(valid_indices)
            # X_valid = X_TRAIN_RAW[valid_indices]
            # y_valid = LABELS_TRAIN[valid_indices]

        X_valid = X_TEST_RAW
        y_valid = LABELS_TEST
        scheduler = [(0, options['n_models'], 1)]
        model_class = partial(ConvNet_wide, n_channels, net_norm='none', im_size=(
            im_size, im_size), k=2, chopped_head=True)
        # n_iters = 100000 if not options['coreset'] else 1
        n_iters = self.n_iters
        whitening_mat = get_zca_matrix(X_TRAIN_RAW, reg_coef=0.1)
        from_loader = False
        X_init = make_coreset(
            X_TRAIN_RAW, LABELS_TRAIN, self.ipc, n_classes, options['init_strategy'], seed=options['seed'], device=self.server_gpu_id)
        self.distill_dataset(X_TRAIN_RAW, LABELS_TRAIN,
                             model_class, options['lr'], 8, options['n_batches'], iters=n_iters,
                             ga_steps=options['ga_steps'], platt=options['platt'],
                             schedule=scheduler, save_location=self.save_path, samples_per_class=self.ipc, n_classes=n_classes,
                             learn_labels=options['learn_labels'], batch_size=1280, X_valid=X_valid, y_valid=y_valid,
                             n_channels=n_channels, im_size=im_size, X_init=X_init, jit=options['jit'], seed=options['seed'], corruption=options['corruption'], whitening_mat=whitening_mat, from_loader=from_loader)


def get_inds_from_labels(images, labels, number_of_classes=100):
    classes_using = [{'class': 0, 'n_per_class': number_of_classes},
                     {'class': 1, 'n_per_class': number_of_classes},
                     ]
    inds = np.concatenate([
        np.random.choice(np.where(labels == c['class'])[
            0], c['n_per_class'], replace=False)
        for c in classes_using
    ])

    return images[inds], labels[inds]


if __name__ == '__main__':
    options = {
        'coreset': False, 'corruption': 0, 'dataset': 'cifar10', 'ga_steps': 1,
        'init_strategy': 'random', 'jit': 0.005, 'learn_labels': True, 'lr': 0.001,
        'n_batches': 4, 'n_models': 8, 'path_dataset': '../data', 'platt': True,
        'samples_per_class': 10, 'save_path': '/mnt/disk1/hieunm/dungntuan/finalize/data-distill-fl/benchmark/cifar10/data', 'seed': 0
    }
    n_channels = 3
    im_size = 32
    X_train, y_train, X_test, y_test = get_cifar10(
        output_channels=n_channels, image_size=im_size, path_dataset=options['path_dataset'])
    X_train, y_train = get_inds_from_labels(
        X_train, y_train, number_of_classes=50)
    X_test, y_test = get_inds_from_labels(
        X_test, y_test, number_of_classes=50)
    breakpoint()
    distill = RFAD_Distillation(
        ipc=10, server_gpu_id='0', save_path='./result', dataset='cifar10', n_iters=10000)
    distill.distill(X_TRAIN_RAW=X_train, LABELS_TRAIN=y_train,
                    X_TEST_RAW=X_test, LABELS_TEST=y_test, options=options)
