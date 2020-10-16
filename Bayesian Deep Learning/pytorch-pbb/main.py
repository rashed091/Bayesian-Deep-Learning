import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import mnist_dataset
import models
import bounds

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--ntrain', type=int, default=60000)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()
cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


def train_noisy(model, train_loader, optimizer, epoch, prior_means, prior_sigmas,
                objective='pbb', penalty=1):
    model.train()

    def compute_bound(pred_err, means, sigmas):
        if objective == 'dziugaite':
            bound, kl = bounds.dziugaite(pred_err, means, sigmas, prior_means, prior_sigmas)
            return bound
        elif objective == 'kl':
            kl, _, _ = bounds.kl_to_prior(means, sigmas, prior_means, prior_sigmas)
            return pred_err + penalty * kl
        elif objective == 'pbb':
            return bounds.f_quad(pred_err, means, sigmas, prior_means, prior_sigmas)
        else:
            assert False

    mean_pred_err = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # get data
        data, target = data.to(DEVICE), target.to(DEVICE)

        # calculate loss
        output = model(data)
        pred_err = (1 / (-np.log(model.PMIN))) * F.nll_loss(output, target)
        mean_pred_err += pred_err.item()

        means = model.get_means()
        sigmas = model.get_sigmas()

        bound = compute_bound(pred_err, means, sigmas)
        # kl = bounds.kl_to_prior(means, sigmas, prior_means, prior_sigmas)[0]
        # print(pred_err.item(), kl.item())
        # # if kl.item() > 1000:
        # import ipdb; ipdb.set_trace()

        # take step
        optimizer.zero_grad()
        bound.backward()
        optimizer.step()

    batch_idx += 1
    mean_pred_err /= batch_idx
    means = model.get_means()
    sigmas = model.get_sigmas()
    avg_surr_bound = compute_bound(mean_pred_err, means, sigmas)
    return 'Train Epoch: {} Batch: {} \t LR: {:.3e} \t Log loss: {:.6f}\t Surrogate bound: {:.6f}'.format(
        epoch, batch_idx, optimizer.param_groups[0]['lr'], mean_pred_err, avg_surr_bound)


def train_lagrangian(model, lambda_param, train_loader, optimizer,
                     lambda_optimizer, epoch, prior_means, prior_sigmas, max_error=0.1):
    model.train()

    def main_loss(log_loss, means, sigmas):
        kl, _, _ = bounds.kl_to_prior(means, sigmas, prior_means, prior_sigmas)
        return lambda_param() * log_loss + kl / (len(train_loader) * args.batch_size)

    def lagrangian_loss(pred_err):
        return max(min(max_error - pred_err, 0.2), -0.2)

    samples = 0
    mean_log_loss = 0
    mean_lambda_loss = 0
    correct = 0

    # optimize the main loss
    for batch_idx, (data, target) in enumerate(train_loader):
        # get data
        data, target = data.to(DEVICE), target.to(DEVICE)

        # calculate loss
        output = model(data)
        log_loss = F.nll_loss(output, target)

        # calculate accuracy
        # get the index of the max log-probability
        pred = output.argmax(dim=1, keepdim=True)
        batch_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += batch_correct

        means = model.get_means()
        sigmas = model.get_sigmas()

        loss = main_loss(log_loss, means, sigmas)

        # take step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # take step on lambda
        lambda_optimizer.zero_grad()
        lambda_loss = lagrangian_loss(1 - (batch_correct / len(data)))
        lambda_pregrad = lambda_loss * lambda_param()
        lambda_pregrad.backward()
        lambda_optimizer.step()

        mean_log_loss += log_loss.item()
        mean_lambda_loss += lambda_loss
        samples += len(data)
    # accuracy = correct / samples
    batch_idx += 1

    # logging
    mean_log_loss /= batch_idx
    mean_lambda_loss /= batch_idx
    means = model.get_means()
    sigmas = model.get_sigmas()
    surrogate_pbb_bound = bounds.f_quad(
        mean_log_loss, means, sigmas, prior_means, prior_sigmas)
    return ('Train Epoch: {} Batch: {} \t LR: {:.3e} \t Log loss: {:.6f}\t '
            'Lambda: {:.6f}\t Lambda loss: {:.6f}\t Surrogate PBB bound: {:.6f}').format(
        epoch, batch_idx, optimizer.param_groups[0]['lr'], mean_log_loss,
        lambda_param().item(), mean_lambda_loss, surrogate_pbb_bound)


def eval_noisy(model, eval_loader, prior_means, prior_sigmas, name, length):
    model.eval()
    test_loss = torch.zeros(1).to(DEVICE)
    correct = 0
    samples = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum')
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            samples += len(data)

            if samples >= length:
                break

    test_loss /= samples
    incorrect = samples - correct

    means = model.get_means()
    sigmas = model.get_sigmas()

    test_risk = incorrect / samples
    dz_bound, _ = bounds.dziugaite(test_risk, means, sigmas, prior_means, prior_sigmas, n=length)
    pbb_bound = bounds.f_quad(test_risk, means, sigmas, prior_means, prior_sigmas, n=length)
    kl, mean_term, sigma_term = bounds.kl_to_prior(means, sigmas, prior_means, prior_sigmas)

    # print("Means distance: {:.4e}, ")

    return ('{} set: Error: {}/{} ({:.0f}%), Log loss: {:.6f}, Dziugaite bound: {:.6f}, PBB bound: {:.6f}, '
            'KL: {:.6f}, Mean term: {:.6f}, Sigma term: {:.6f}').format(
        name,
        incorrect, samples, 100. * incorrect / samples,
        test_loss.item(), dz_bound.item(), pbb_bound.item(),
        kl, mean_term, sigma_term)


kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    mnist_dataset.MNIST('../data', train=True, download=True, n_examples=args.ntrain,
                        transform=transforms.Compose([
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    mnist_dataset.MNIST('../data', train=False, download=True, n_examples=10000,
                        transform=transforms.Compose([
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


prior_sigma = 3e-2

lambda_param = models.Lagrangian(1e-1).to(DEVICE)
lambda_optimizer = optim.SGD(lambda_param.parameters(), lr=1e-2)
model = models.NoisyNet(prior_sigma, per_sample=False,
                        clipping='hard').to(DEVICE)
# model = models.SmallNoisyNet(prior_sigma, per_sample=False,
#                              clipping='hard').to(DEVICE)
# model = models.TinyNoisyNet(prior_sigma, per_sample=False,
#                             clipping='hard').to(DEVICE)

prior_means = [p.clone().detach() for p in model.get_means()]
prior_sigmas = [p.clone().detach() for p in model.get_sigmas()]

optimizer = optim.SGD(model.parameters(), lr=5e-3, momentum=0.95)
# optimizer = optim.Adadelta(model.parameters(), lr=1e-1)
# optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
# scheduler = StepLR(optimizer, step_size=100, gamma=0.3)

train_eval_str = eval_noisy(model, train_loader, prior_means, prior_sigmas, "Train", args.ntrain)
test_eval_str = eval_noisy(model, test_loader, prior_means, prior_sigmas, "Test", 10000)
print(train_eval_str)
print(test_eval_str)
print()

for epoch in range(0, 10000):
    train_str = train_noisy(model, train_loader, optimizer,
                            epoch, prior_means, prior_sigmas, objective='pbb')
    # train_str = train_lagrangian(model, lambda_param, train_loader, optimizer,
    #                              lambda_optimizer, epoch, prior_means, prior_sigmas, max_error=0.01)

    if epoch % 1 == 0:
        train_eval_str = eval_noisy(
            model, train_loader, prior_means, prior_sigmas, "Train", args.ntrain)
        test_eval_str = eval_noisy(
            model, test_loader, prior_means, prior_sigmas, "Test", 10000)
        print("Epoch:", epoch)
        print(train_str)
        print(train_eval_str)
        print(test_eval_str)
        print()
#     scheduler.step()
