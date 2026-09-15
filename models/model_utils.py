import torch
import torch.nn.functional as F
import numpy as np


def cosine_sim(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return cos_sim_fn(x[..., None], z.T[None, ...])


def cos_dist(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return (1 - cos_sim_fn(x[..., None], z.T[None, ...])) / 2


def linear_sim(x, z):
    return x @ z.T


def l2_dist(x, z):
    dist_squared = (x**2).sum() + (z**2).sum() - 2 * linear_sim(x, z)
    return torch.clamp(dist_squared, min=0).sqrt()


def cos_loglikelihood(x, z, gamma=0.1, z_dim=1):
    cos_sim = cosine_sim(x, z)
    probs = F.softmax(cos_sim / gamma, dim=z_dim)
    return torch.log(probs)


def unique_softmax(sim, labels, gamma=1, dim=0):
    assert sim.shape[0] == labels.shape[0]
    labels = labels.detach().cpu().numpy()
    unique_labels, unique_index, unique_inverse_index = np.unique(labels, return_index=True, return_inverse=True)
    unique_sim = sim[unique_index]
    unique_softmax_sim = torch.nn.functional.softmax(unique_sim / gamma, dim=dim)
    softmax_sim = unique_softmax_sim[unique_inverse_index]
    return softmax_sim
