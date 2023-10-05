import copy
from math import ceil
from warnings import catch_warnings
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from numpy import random
from torch.autograd import grad
import torch.nn.functional as F
from torch.optim import SGD


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def inner_loss_func(preference, output, target):
    return preference * F.cross_entropy(output, target)


def cross_entropy(output, target):
    return F.cross_entropy(output, target)


class Client():
    """
    * @param args
    * @param client_id
    * @param net
    * @param dataset
    * @param idxs
    * @param preference
    """

    def __init__(self, args, client_id, net, train_dataset=None, val_dataset=None,
                 train_idxs=None, val_idxs=None, preference=None) -> None:
        self.client_id = client_id
        self.args = args
        self.net = net.clone()
        self.init_net = copy.deepcopy(net)
        self.net.zero_grad()
        self.init_net.zero_grad()
        self.beta = 0.1

        self.ldr_train = DataLoader(DatasetSplit(
            train_dataset, train_idxs[client_id]), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_val = DataLoader(DatasetSplit(
            val_dataset, val_idxs[client_id]), batch_size=self.args.local_bs, shuffle=True)

        self.preference = preference.clone()
        self.preference_init = preference.clone()
        self.val_loss = cross_entropy
        self.loss_func = inner_loss_func

    def train_epoch(self):
        """
        Inner optimization, each client trains its local model for E epochs
        """
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        optimizer = SGD(self.net0.parameters(), lr=self.args.ilr)
        self.net0.zero_grad()
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                log_probs = self.net0(images)
                loss = self.loss_func(self.preference, log_probs, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return self.net0.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def maml_train_epoch(self):
        """
        Inner optimization, each client trains its local model for E epochs
        """
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        # optimizer = SGD(self.net0.parameters(), lr=self.args.ilr)
        # self.net0.zero_grad()
        # epoch_loss = []
        for iter in range(self.args.local_ep):
            # batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                train_error = self.loss_func(self.net0(images), labels)
                self.net0.adapt(train_error)
        return self.net0.state_dict(), train_error

    def outer_train(self):
        """
        Outer optimization, each client trains its local preference for E epochs
        """
        self.net0 = copy.deepcopy(self.net)
        self.net0.train()
        optimizer = SGD([nn.Parameter(self.preference)], lr=self.args.olr)
        epoch_loss = []
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            log_probs = self.net0(images)
            loss = self.val_loss(log_probs, labels)
            # loss.backward(retain_graph=True)
            # optimizer.step()
            preference_grad = grad(loss, self.preference, retain_graph=True)[0]
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.net0.zero_grad()
        return self.preference, sum(epoch_loss) / len(epoch_loss)

    def maml_outer_train(self):
        """
        Outer optimization, each client trains its local preference for E epochs
        """
        self.net0 = copy.deepcopy(self.net)
        # self.net0.train()
        # optimizer = SGD([nn.Parameter(self.preference)], lr=self.args.olr)
        epoch_loss = []
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(self.ldr_val):
            images, labels = images.to(
                self.args.device), labels.to(self.args.device)
            log_probs = self.net0(images)
            loss = self.val_loss(log_probs, labels)
            # loss.backward(retain_graph=True)
            # optimizer.step()
            # preference_grad = grad(loss, self.preference, retain_graph=True)[0]
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)
