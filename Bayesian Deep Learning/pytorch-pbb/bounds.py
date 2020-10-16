import torch
import math
import numpy as np
import torch.distributions as td


def kl_to_prior(means, sigmas, prior_means, prior_sigmas):
    means = torch.cat([m.flatten() for m in means])
    prior_means = torch.cat([m.flatten() for m in prior_means])

    sigmas = torch.cat([s.flatten() for s in sigmas])
    prior_sigmas = torch.cat([s.flatten() for s in prior_sigmas])

    mu_diff = means - prior_means
    mu_term = (mu_diff**2 / prior_sigmas**2).sum()

    sigma_tr_term = (sigmas**2 / (prior_sigmas**2)).sum()

    # take the log first to make everything linear,
    # then do the sum-of-differences instead of the difference-of-sums
    # to make the numerics better
    log_det_term = 2 * (torch.log(prior_sigmas) - torch.log(sigmas)).sum()

    k = means.shape[0]

    # group calculation of (sigma_tr + log_det - k) to avoid numerical issues;
    # each term is large but the difference is ~0
    sigma_term = sigma_tr_term - k + log_det_term
    kl = 0.5 * (mu_term + sigma_term)

    return kl, 0.5 * mu_term, 0.5 * sigma_term


def td_kl_to_prior(means, sigmas, prior_means, prior_sigmas):
    means = torch.cat([m.flatten() for m in means])
    prior_means = torch.cat([m.flatten() for m in prior_means])

    sigmas = torch.cat([s.flatten() for s in sigmas])
    prior_sigmas = torch.cat([s.flatten() for s in prior_sigmas])

    q = td.MultivariateNormal(means, torch.diag(sigmas))
    p = td.MultivariateNormal(prior_means, torch.diag(sigmas))
    return td.kl_divergence(q, p).sum(), torch.tensor(0.), torch.tensor(0.)


def maria_kl_to_prior(means, sigmas, prior_means, prior_sigmas):
    means = torch.cat([m.flatten() for m in means])
    prior_means = torch.cat([m.flatten() for m in prior_means])
    sigmas = torch.cat([s.flatten() for s in sigmas])
    prior_sigmas = torch.cat([s.flatten() for s in prior_sigmas])

    var1 = sigmas ** 2
    var0 = prior_sigmas ** 2

    aux1 = torch.log(torch.div(var0, var1))
    aux2 = torch.div(
        torch.pow(means - prior_means, 2), var0)
    aux3 = torch.div(var1, var0)
    kl_div = torch.mul(aux1 + aux2 + aux3 - 1, 0.5).sum()

    return kl_div, torch.tensor(0.), torch.tensor(0.)


def f_quad(empirical_risk,
           means, sigmas, prior_means, prior_sigmas,
           delta=(1 - 0.95), n=60000):
    kl, _, _ = kl_to_prior(means, sigmas, prior_means, prior_sigmas)
    kl_ratio = (kl + np.log(2 * np.sqrt(n) / delta)) / (2 * n)
    return ((empirical_risk + kl_ratio)**0.5 + (kl_ratio)**0.5)**2


def dziugaite(empirical_risk, means, sigmas, prior_means, prior_sigmas,
              delta=(1 - 0.95), n=60000):
    kl, _, _ = kl_to_prior(means, sigmas, prior_means, prior_sigmas)
    gap = ((kl + math.log(n / delta)) / (2 * (n - 1))) ** 0.5
    return empirical_risk + gap, kl


def KLdiv(pbar, p):
    return pbar * np.log(pbar / p) + (1 - pbar) * np.log((1 - pbar) / (1 - p))


def KLdiv_prime(pbar, p):
    return (1 - pbar) / (1 - p) - pbar / p


def Newt(p, q, c):
    newp = p - (KLdiv(q, p) - c) / KLdiv_prime(q, p)
    return newp


def approximate_BPAC(train_accur, B_init, niter=5):
    """Input B_init should be (KL + log_term) / n"""
    B_RE = 2 * B_init ** 2
    A = 1 - train_accur
    B_next = B_init + A
    if B_next > 1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next, A, B_RE)
    return B_next
