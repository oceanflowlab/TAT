import numpy as np
import torch
import random
import math

from itertools import product
from torch import log, exp
import torch.nn.functional as F
from models.model_utils import unique_softmax


device = "cuda" if torch.cuda.is_available() else "cpu"


def calibrated_pairwise_scores(sample, gamma_xz):
    """Return Eq. (27) scores at the Drop-DTW gamma used by this call.

    ``pairwise_scores`` are stored at the training/reference gamma.  RAW
    Drop-DTW divides its logits by the caller's ``gamma_xz`` both during
    training and evaluation.  Applying the same ratio here prevents the
    guided branch from silently bypassing a validation-time gamma change.
    The positive rescaling changes confidence/drop calibration, not ranking.
    """
    scores = sample.get('pairwise_scores')
    if scores is None:
        return None
    reference_gamma = sample.get('pairwise_score_reference_gamma')
    if reference_gamma is None:
        return scores
    reference_gamma = float(reference_gamma)
    gamma_xz = float(gamma_xz)
    if reference_gamma <= 0 or gamma_xz <= 0:
        raise ValueError("Pairwise score reference/current gamma must be positive")
    return scores * (reference_gamma / gamma_xz)


def compute_all_costs(
    sample,
    distractor,
    gamma_xz,
    drop_cost_type,
    keep_percentile,
    l2_nomalize=False,
    distinct_step_occurrences=False,
):
    """This function computes pairwise match and individual drop costs used in Drop-DTW

    Parameters
    __________

    sample: dict
        sample dictionary
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    distractor: torch.tensor of size [d] or None
        Background class prototype. Only used if the drop cost is learnable.
    drop_cost_type: str
        The type of drop cost definition, i.g., learnable or logits percentile.
    keep_percentile: float in [0, 1]
        if drop_cost_type == 'logit', defines drop (keep) cost threshold as logits percentile
    l2_normalize: bool
        wheather to normalize clip and step features before computing the costs
    """

    labels = sample['step_ids']
    step_features, frame_features = sample['step_features'], sample['frame_features']
    if l2_nomalize:
        frame_features = F.normalize(frame_features, p=2, dim=1)
        step_features = F.normalize(step_features, p=2, dim=1)
    # Memory-guided samples may provide Eq. (27)'s prediction matrix P_u.
    # It is already cosine / temperature, so do not recompute a dot product
    # or apply gamma_xz a second time.
    pairwise_scores = calibrated_pairwise_scores(sample, gamma_xz)
    if pairwise_scores is not None:
        expected_shape = (step_features.shape[0], frame_features.shape[0])
        if pairwise_scores.shape != expected_shape:
            raise ValueError(
                f"pairwise_scores has shape {tuple(pairwise_scores.shape)}, "
                f"expected {expected_shape}"
            )
        sim = pairwise_scores
        score_temperature = 1.0
    else:
        sim = step_features @ frame_features.T
        score_temperature = gamma_xz

    if distinct_step_occurrences:
        unique_index = np.arange(labels.numel())
        unique_inverse_index = np.arange(labels.numel())
    else:
        _, unique_index, unique_inverse_index = np.unique(
            labels.detach().cpu().numpy(), return_index=True, return_inverse=True
        )
    unique_sim = sim[unique_index]

    if drop_cost_type == 'logit':
        k = max([1, int(torch.numel(unique_sim) * keep_percentile)])
        baseline_logit = torch.topk(unique_sim.reshape([-1]), k).values[-1].detach()
        baseline_logits = baseline_logit.repeat([1, unique_sim.shape[1]])  # making it of shape [1, N]
        sims_ext = torch.cat([unique_sim, baseline_logits], dim=0)
    elif drop_cost_type == 'learn':
        if pairwise_scores is not None:
            tau = float(sample['pairwise_score_temperature'])
            reference_gamma = sample.get('pairwise_score_reference_gamma')
            if reference_gamma is not None:
                tau *= float(gamma_xz) / float(reference_gamma)
            distractor_sim = (
                F.normalize(frame_features, p=2, dim=1)
                @ F.normalize(distractor, p=2, dim=0)
            ) / tau
        else:
            distractor_sim = frame_features @ distractor
        sims_ext = torch.cat([unique_sim, distractor_sim[None, :]], dim=0)
    else:
        assert False, f"No such drop mode {drop_cost_type}"

    unique_softmax_sims = torch.nn.functional.softmax(
        sims_ext / score_temperature, dim=0
    )
    unique_softmax_sim, drop_probs = unique_softmax_sims[:-1], unique_softmax_sims[-1]
    matching_probs = unique_softmax_sim[unique_inverse_index]
    zx_costs = -torch.log(matching_probs + 1e-5)
    drop_costs = -torch.log(drop_probs + 1e-5)
    return zx_costs, drop_costs, drop_probs


class VarTable():
    def __init__(self, dims, dtype=torch.float, device=device):
        self.dims = dims
        d1, d2, d_rest = dims[0], dims[1], dims[2:]

        self.vars = []
        for i in range(d1):
            self.vars.append([])
            for j in range(d2):
                var = torch.zeros(d_rest).to(dtype).to(device)
                self.vars[i].append(var)

    def __getitem__(self, pos):
        i, j = pos
        return self.vars[i][j]

    def __setitem__(self, pos, new_val):
        i, j = pos
        if self.vars[i][j].sum() != 0:
            assert False, "This cell has already been assigned. There must be a bug somwhere."
        else:
            self.vars[i][j] = self.vars[i][j] + new_val

    def show(self):
        device, dtype = self[0, 0].device, self[0, 0].dtype
        mat = torch.zeros((self.d1, self.d2, self.d3)).to().to(dtype).to(device)
        for dims in product([range(d) for d in self.dims]):
            i, j, rest = dims[0], dims[1], dims[2:]
            mat[dims] = self[i, j][rest]
        return mat


def minGamma(inputs, gamma=1, keepdim=True):
    """ continuous relaxation of min defined in the D3TW paper"""
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minG = inputs.min(dim=0, keepdim=keepdim)
    else:
        # log-sum-exp stabilization trick
        zi = (-inputs / gamma)
        max_zi = zi.max()
        log_sum_G = max_zi + log(exp(zi - max_zi).sum(dim=0, keepdim=keepdim) + 1e-5)
        minG = -gamma * log_sum_G
    return minG


def minProb(inputs, gamma=1, keepdim=True):
    if type(inputs) == list:
        if inputs[0].shape[0] == 1:
            inputs = torch.cat(inputs)
        else:
            inputs = torch.stack(inputs, dim=0)

    if gamma == 0:
        minP = inputs.min(dim=0, keepdim=keepdim)
    else:
        probs = F.softmax(-inputs / gamma, dim=0)
        minP = (probs * inputs).sum(dim=0, keepdim=keepdim) 
    return minP


def traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
