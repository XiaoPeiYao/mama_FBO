import copy
from client import Client
from Fed import *


class clientManage():
    def __init__(self, args, net_glob, client_idx, train_dataset,
                 val_dataset, train_dict_users, val_dict_users, preference) -> None:
        self.args = args
        self.net_glob = net_glob
        self.client_idx = client_idx
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dict_users = train_dict_users
        self.val_dict_users = val_dict_users
        self.preference = preference

    def inner(self):
        param_glob = self.net_glob.state_dict()
        if self.args.all_clients:
            param_locals = [param_glob for i in range(self.args.num_users)]
        else:
            param_locals = []
        loss_locals = []
        client_locals = []
        for idx in self.client_idx:
            client = Client(self.args, idx, copy.deepcopy(self.net_glob), self.train_dataset,
                            self.val_dataset, self.train_dict_users, self.val_dict_users, self.preference[idx])
            client_locals.append(client)
        for client in client_locals:
            param, loss = client.train_epoch()
            if self.args.all_clients:
                param_locals[idx] = copy.deepcopy(param)
            else:
                param_locals.append(copy.deepcopy(param))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        param_glob = FedAvg(param_locals)
        # dx_dp = torch.autograd.grad(list(param_glob.values()), self.preference, create_graph=True)
        # copy weight to net_glob
        self.net_glob.load_state_dict(param_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        return param_glob, loss_avg, client_locals


    def maml_inner(self):
        param_glob = self.net_glob.state_dict()
        if self.args.all_clients:
            param_locals = [param_glob for i in range(self.args.num_users)]
        else:
            param_locals = []
        loss_locals = []
        client_locals = []
        for idx in self.client_idx:
            client = Client(self.args, idx, self.net_glob, self.train_dataset,
                            self.val_dataset, self.train_dict_users, self.val_dict_users, self.preference[idx])
            client_locals.append(client)
        for client in client_locals:
            param, loss = client.maml_train_epoch()
            if self.args.all_clients:
                param_locals[idx] = copy.deepcopy(param)
            else:
                param_locals.append(copy.deepcopy(param))
            loss_locals.append(loss)
        # update global weights
        param_glob = FedAvg(param_locals)
        # dx_dp = torch.autograd.grad(list(param_glob.values()), self.preference, create_graph=True)
        # copy weight to net_glob
        self.net_glob.load_state_dict(param_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        return param_glob, loss_avg, client_locals
    def outer(self, client_locals):
        list_loss = []
        for client in client_locals:
            self.preference[client.client_id], val_loss = client.outer_train()
            list_loss.append(val_loss)
        return self.preference, sum(list_loss) / len(list_loss)

    def maml_outer(self, client_locals):
        list_loss = []
        for client in client_locals:
            val_loss = client.maml_outer_train()
            list_loss.append(val_loss)
        return sum(list_loss) / len(list_loss)


