#!/usr/bin/env python3

from options import args_parser
import random
import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from core.clientManage import clientManage
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from utils import get_dataset, average_weights, exp_details
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
import learn2learn as l2l


class Net(nn.Module):
    def __init__(self, ways=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, ways)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1)
    acc = (predictions == targets).sum().float()
    acc /= len(targets)
    return acc.item()


def main(args, device=torch.device("cpu")):
    # transformations = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,)),
    #     lambda x: x.view(1, 28, 28),
    # ])
    #
    # mnist_train = l2l.data.MetaDataset(MNIST(download_location,
    #                                          train=True,
    #                                          download=True,
    #                                          transform=transformations))
    #
    # train_tasks = l2l.data.Taskset(
    #     mnist_train,
    #     task_transforms=[
    #         l2l.data.transforms.NWays(mnist_train, ways),
    #         l2l.data.transforms.KShots(mnist_train, 2*shots),
    #         l2l.data.transforms.LoadData(mnist_train),
    #         l2l.data.transforms.RemapLabels(mnist_train),
    #         l2l.data.transforms.ConsecutiveLabels(mnist_train),
    #     ],
    #     num_tasks=1000,
    # )
    train_dataset, val_dataset, test_dataset, train_user_groups, val_user_groups = get_dataset(args)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # preference for different clients and do projection
    preference = torch.rand(100, requires_grad=True)
    preference = torch.nn.functional.softmax(preference)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss_list, train_acc_list = [], []
    val_loss_list, val_acc_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    upper_model = l2l.algorithms.MAML(global_model, lr=args.olr)
    upper_opt = optim.Adam(preference, lr=args.olr)

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')
        # inner_model = upper_model.clone()
        # inner_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        clients_manage = clientManage(args, upper_model, idxs_users, train_dataset,
                                      val_dataset, train_user_groups, val_user_groups, preference)
        # inner update
        param_glob, train_loss, client_locals = clients_manage.maml_inner()
        # update global weights
        upper_model.load_state_dict(param_glob)
        # outer loop, update preference
        val_loss = clients_manage.maml_outer(client_locals)
        upper_opt.zero_grad()
        val_loss.backward()
        upper_opt.step()
        # projection
        for param_group in upper_opt.param_groups:
            for param in param_group['params']:
                param.data = torch.nn.functional.softmax(param.data.flatten(), dim=0).view(param.data.size())


if __name__ == '__main__':
    args = args_parser()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    main(args=args, device=device)
