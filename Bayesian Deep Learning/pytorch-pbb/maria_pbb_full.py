import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# make the import work when running the main file in this package
# or when importing the whole package
try:
    from .truncnormal import trunc_normal_
    from .dataset_utils import DatasetCache
except ImportError:
    from truncnormal import trunc_normal_
    from dataset_utils import DatasetCache


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"

LOADER_KWARGS = {'num_workers': 0,
                 'pin_memory': True} if torch.cuda.is_available() else {}

# print(torch.cuda.is_available())

# Load and transform the dataset
BATCH_SIZE = 250
# TODO: leave 10000 left for validation
transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
     ])

train_loader = torch.utils.data.DataLoader(
    DatasetCache(datasets.MNIST(
        'mnist-data/', train=True, download=True,
        transform=transform)),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    DatasetCache(datasets.MNIST(
        'mnist-data/', train=False, download=True,
        transform=transform)),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)


# TODO: if we want to test simply dividing by 255 use this:
#train_loader.dataset.data = train_loader.dataset.data.float() / 255
#test_loader.dataset.data = test_loader.dataset.data.float() / 255
# and comment the previous transforms.Normalize

TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
CLASSES = 10

# 1 epoch = 20.000 iterations
TRAIN_EPOCHS = 3501
NUM_BATCHES = len(train_loader)

# Best hyperparameters
SIGMAPRIOR = 3e-2
RHO_PRIOR = math.log(math.exp(SIGMAPRIOR)-1.0)
PMIN = 1e-3
DELTA = 0.05
LEARNING_RATE = 5e-3
MOMENTUM = 0.95


class Gaussian(nn.Module):
    def __init__(self, mu, rho, fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)

    @property
    def sigma(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians
        # (please refer to Appendix A of the paper)
        # b0 is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()

        return kl_div


class ProbLinear(nn.Module):
    # Our network will be made of probabilistic linear layers
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # Initialise Q weight means with truncated normal,
        # initialise Q weight scales from RHO_PRIOR
        # check tensorflow truncated normal initialisation
        weights_mu_init = trunc_normal_(torch.Tensor(
            out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        weights_rho_init = torch.ones(out_features, in_features) * RHO_PRIOR
        self.weight = Gaussian(weights_mu_init.clone(),
                               weights_rho_init.clone(), fixed=False)

        # Initialise Q bias means with truncated normal,
        # initialise Q bias rhos from RHO_PRIOR
        bias_mu_init = torch.zeros(out_features)
        bias_rho_init = torch.ones(out_features) * RHO_PRIOR
        self.bias = Gaussian(bias_mu_init.clone(),
                             bias_rho_init.clone(), fixed=False)

        # Set prior Q_0 using random initialisation and RHO_PRIOR
        self.weight_prior = Gaussian(
            weights_mu_init.clone(), weights_rho_init.clone(), fixed=True)
        self.bias_prior = Gaussian(
            bias_mu_init.clone(), bias_rho_init.clone(), fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)


class ProbNetwork(nn.Module):
    def __init__(self):
        # initialise our network
        super().__init__()
        self.l1 = ProbLinear(28*28, 600)
        self.l2 = ProbLinear(600, 600)
        self.l3 = ProbLinear(600, 600)
        self.l4 = ProbLinear(600, 10)

    def forward(self, x, sample=False):
        # forward pass for the network
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = self.output_transform(self.l4(x, sample))
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div

    def output_transform(self, x):
        # lower bound output prob
        output = F.log_softmax(x, dim=1)
        return torch.clamp(output, np.log(PMIN))

    def fquad(self, input, target):
        # Implementation of fquad training objective

        # we clamp the outputs of the softmax with PMIN
        # note that we sample once per batch
        outputs = self(input, sample=True)
        # we compute the KL divergence between Q and Q_0
        kl_div = self.compute_kl()
        # compute cross_entropy
        cross_entropy = (1/(-np.log(PMIN))) * F.nll_loss(outputs, target)
        repeated_kl_ratio = torch.div(
            kl_div + np.log((2*np.sqrt(TRAIN_SIZE))/DELTA), 2*TRAIN_SIZE)
        first_term = torch.sqrt(cross_entropy + repeated_kl_ratio)
        second_term = torch.sqrt(repeated_kl_ratio)
        # compute training objective
        loss = torch.pow(first_term + second_term, 2)
        return loss,  kl_div/TRAIN_SIZE, outputs, cross_entropy

    def fquad_1_0(self, risk):
        # we compute the KL divergence between Q and Q_0
        kl_div = self.compute_kl()
        repeated_kl_ratio = torch.div(
            kl_div + np.log((2*np.sqrt(TRAIN_SIZE))/DELTA), 2*TRAIN_SIZE)
        first_term = torch.sqrt(risk + repeated_kl_ratio)
        second_term = torch.sqrt(repeated_kl_ratio)
        # compute training objective
        risk_bound = torch.pow(first_term + second_term, 2)
        return risk_bound


net = ProbNetwork().to(DEVICE)


def train(net, optimizer, epoch):
    # train and report training metrics
    net.train()
    total, correct, avgbound, avgkl, avgloss = 0.0, 0.0, 0.0, 0.0, 0.0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        bound, kl, output, loss = net.fquad(data, target)
        bound.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        avgbound = avgbound + bound.detach()
        avgkl = avgkl + kl.detach()
        avgloss = avgloss + loss.detach()
    # show the average loss and KL during the epoch
    print(f"-Epoch {epoch :.5f}, RUB: {avgbound/batch_id :.5f}, KL/n: {avgkl/batch_id :.5f}, Train loss: {avgloss/batch_id :.5f}, Train Acc:  {correct/total * 100:.5f}")


def compute_full_bound():
    cross_entropy, correct, total = 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            bound, kl, output, loss = net.fquad(data, target)
            cross_entropy += loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    cross_entropy /= batch_id
    surrogate_bound = net.fquad_1_0(cross_entropy)
    accuracy = correct / total
    risk_bound = net.fquad_1_0(1 - accuracy)
    print(
        f"Train accuracy: {accuracy :.4f}, Bound: {risk_bound :.4f}, Surrogate bound: {surrogate_bound :.4f}")


def test():
    # compute posterior mean test accuracy
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = net(data)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('Posterior Mean Test Accuracy: {}/{}'.format(correct, TEST_SIZE))


if __name__ == '__main__':
    # set optimiser, train and output test accuracy every 10 epochs
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    for epoch in range(TRAIN_EPOCHS):
        train(net, optimizer, epoch)
        compute_full_bound()
        if ((epoch+1) % 10 == 0):
            test()
