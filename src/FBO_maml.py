#!/usr/bin/env python3

from options import args_parser
import random
import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from clientManage import clientManage
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from utils import get_dataset, average_weights, exp_details
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
import learn2learn as l2l


def main(args, device=torch.device("cpu")):
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
    upper_opt = optim.Adam([nn.Parameter(preference)], lr=args.olr)

    for epoch in range(args.epochs):
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
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args=args, device=device)
