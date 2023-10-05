#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from core.clientManage import clientManage
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

np.random.seed(42)


# Euclidean projection
# def euclidean_proj_simplex(v, s=1):
#     """ Compute the Euclidean projection on a positive simplex
#     Solves the optimisation problem (using the algorithm from [1]):
#         min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
#     Parameters
#     ----------
#     v: (n,) numpy array,
#        n-dimensional vector to project
#     s: int, optional, default: 1,
#        radius of the simplex
#     Returns
#     -------
#     w: (n,) numpy array,
#        Euclidean projection of v on the simplex
#     Notes
#     -----
#     The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
#     Better alternatives exist for high-dimensional sparse vectors (cf. [1])
#     However, this implementation still easily scales to millions of dimensions.
#     References
#     ----------
#     [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
#         John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
#         International Conference on Machine Learning (ICML 2008)
#         http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
#     [2] Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application
#         Weiran Wang, Miguel Á. Carreira-Perpiñán. arXiv:1309.1541
#         https://arxiv.org/pdf/1309.1541.pdf
#     [3] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246#file-simplex_projection-py
#     """
#     assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#     v = v.astype(np.float64)
#     n = v.shape[0]  # will raise ValueError if v is not 1-D
#     # check if we are already on the simplex
#     if v.sum() == s and np.alltrue(v >= 0):
#         # best projection: itself!
#         return v
#     # get the array of cumulative sums of a sorted (decreasing) copy of v
#     u = np.sort(v)[::-1]
#     cssv = np.cumsum(u)
#     # get the number of > 0 components of the optimal solution
#     rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
#     # compute the Lagrange multiplier associated to the simplex constraint
#     theta = float(cssv[rho] - s) / (rho + 1)
#     # compute the projection by thresholding v using theta
#     w = (v - theta).clip(min=0)
#     return w


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # if args.gpu:
    # torch.cuda.set_device(args.gpu_id)
    device = torch.device('mps') if args.gpu else 'cpu'

    # load dataset and user groups
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

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        clients_manage = clientManage(args, global_model, idxs_users, train_dataset,
                                      val_dataset, train_user_groups, val_user_groups, preference)
        # inner steps with training data
        param_glob, train_loss, client_locals = clients_manage.inner()
        # update global weights
        global_model.load_state_dict(param_glob)
        # outer loop, update preference
        preference, val_loss = clients_manage.outer(client_locals)
        # projection
        preference = torch.nn.functional.softmax(preference)
        # print global training loss & validation loss after every 'i' round
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {train_loss}')
            print('Validation Loss: {:.2f}% \n'.format(val_loss))

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # list_acc, list_loss = [], []
        # global_model.eval()
        # for idx in idxs_users:
        #     local_model = LocalUpdate(args=args, dataset=val_dataset,
        #                               idxs=val_user_groups[idx], logger=logger)
        #     preference[idx], val_acc = local_model.update_preference(
        #         preference=preference[idx], model=global_model, global_round=epoch)
        #     list_acc.append(val_acc)
        #     # list_loss.append(loss)
        # val_acc_list.append(sum(list_acc) / len(list_acc))
        # # projection
        # preference = torch.nn.functional.softmax(preference)
        #
        # # print global training loss & val_acc after every 'i' round
        # if (epoch + 1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
        #     print('Validation Accuracy: {:.2f}% \n'.format(100 * val_acc_list[-1]))

    # Test inference after completion of training
    global_model.eval()
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and val_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, val_acc_list], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # PLOTTING (optional)

    # matplotlib.use('Agg')

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    # #
    # # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Validation Accuracy vs Communication rounds')
    # plt.plot(range(len(val_acc_list)), val_acc_list, color='k')
    # plt.ylabel('Validation Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
