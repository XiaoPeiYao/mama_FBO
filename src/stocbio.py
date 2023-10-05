import torch
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F


def Stocbio(preference, val_data_loader, train_data_loader, model, reg_f):
    # Gx_gradient and get v_0
    v_0 = torch.zeros_like(model.named_parameters()[1])
    for batch_idx, (images, labels) in enumerate(val_data_loader):
        model.zero_grad()
        log_probs = model(images)
        batch_loss = nn.NLLLoss(log_probs, labels)
        batch_loss.backward()
        batch_grad = model.named_parameters()[1].grad
        v_0 += batch_grad
    v_0 = torch.unsqueeze(torch.reshape(v_0, [-1]), 1).detach()

    # Hessian
    z_list = []
    output = out_f(data_list[1], params)
    Gy_gradient = gradient_gy(args, labels_list[1], params, data_list[1], preference, output, reg_f)

    G_gradient = torch.reshape(params[0], [-1]) - args.eta * torch.reshape(Gy_gradient, [-1])

    for _ in range(args.hessian_q):
        Jacobian = torch.matmul(G_gradient, v_0)
        v_new = torch.autograd.grad(Jacobian, params, retain_graph=True)[0]
        v_0 = torch.unsqueeze(torch.reshape(v_new, [-1]), 1).detach()
        z_list.append(v_0)
    v_Q = args.eta * v_0 + torch.sum(torch.stack(z_list), dim=0)

    # Gyx_gradient
    output = out_f(data_list[2], params)
    Gy_gradient = gradient_gy(args, labels_list[2], params, data_list[2], preference, output, reg_f)
    Gy_gradient = torch.reshape(Gy_gradient, [-1])
    Gyx_gradient = torch.autograd.grad(torch.matmul(Gy_gradient, v_Q.detach()), preference, retain_graph=True)[0]
    outer_update = -Gyx_gradient

    return outer_update


def gradient_gx(args, labels, params, data, output):
    loss = F.cross_entropy(output, labels)
    grad = torch.autograd.grad(loss, params)[0]
    return grad


def gradient_gy(args, labels_cp, params, data, preference, output, reg_f):
    # For MNIST data-hyper cleaning experiments
    loss = F.cross_entropy(output, labels_cp, reduction='none')
    # For NewsGroup l2reg expriments
    # loss = F.cross_entropy(output, labels_cp)
    loss_regu = reg_f(params, preference, loss)
    grad = torch.autograd.grad(loss_regu, params, create_graph=True)[0]
    return grad