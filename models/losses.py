import torch
import torch.nn.functional as F
from torch import log, exp
import numpy as np

from dp.soft_dp import batch_dropDTW
from dp.dp_utils import compute_all_costs
from models.model_utils import unique_softmax


def mil_nce(features_1, features_2, correspondance_mat, eps=1e-8, gamma=1, hard_ratio=1):
    corresp = correspondance_mat.to(torch.float32)
    prod = features_1 @ features_2.T / gamma
    # Compute the exact MIL-NCE ratio in log space.  Subtracting the row
    # maximum before exp is not sufficient: when every positive is far below
    # a negative, the positive numerator can still underflow to zero and turn
    # an otherwise finite loss into +inf.  logsumexp keeps both the value and
    # its gradient finite without clipping the logits or changing the loss.
    positive = corresp > 0
    if not positive.any(dim=1).all():
        raise ValueError("MIL-NCE requires at least one positive per row")
    neg_inf = torch.full_like(corresp, float("-inf"))
    log_corresp = torch.where(positive, corresp.log(), neg_inf)
    log_nominator = torch.logsumexp(prod + log_corresp, dim=1)
    log_denominator = torch.logsumexp(prod, dim=1)
    nll = log_denominator - log_nominator
    if hard_ratio < 1:
        n_hard_examples = int(nll.shape[0] * hard_ratio)
        hard_indices = nll.sort().indices[-n_hard_examples:]
        nll = nll[hard_indices]
    return nll.mean()


def compute_clustering_loss(samples):
    all_pooled_frames, pooled_frames_labels = [], []
    all_step_features, all_step_labels = [], []
    score_only_temperature = None
    for sample in samples:
        step_features, frame_features = sample['step_features'], sample['frame_features']
        pairwise_scores = sample.get('pairwise_scores')
        if pairwise_scores is None:
            raise ValueError("TAT samples must provide pairwise_scores")
        sample_temperature = float(sample['pairwise_score_temperature'])
        if score_only_temperature is None:
            score_only_temperature = sample_temperature
        elif abs(score_only_temperature - sample_temperature) > 1e-12:
            raise ValueError("All TAT samples must share one score temperature")
        step_ids = sample['step_ids']

        unique_step_labels, unique_idxs = [
            torch.from_numpy(t) for t in np.unique(step_ids.detach().cpu().numpy(), return_index=True)]
        unique_step_features = step_features[unique_idxs]  # size [K, d]
        sim = pairwise_scores[unique_idxs]
        frame_weights = F.softmax(sim, dim=1)
        step_pooled_frames = frame_weights @ frame_features  # size [K, d]
        all_pooled_frames.append(step_pooled_frames)
        pooled_frames_labels.append(unique_step_labels)
        all_step_features.append(step_features)
        all_step_labels.append(step_ids)
    all_pooled_frames = torch.cat(all_pooled_frames, dim=0)
    pooled_frames_labels = torch.cat(pooled_frames_labels, dim=0)
    all_step_features = torch.cat(all_step_features, dim=0)
    all_step_labels = torch.cat(all_step_labels, dim=0)
    assert pooled_frames_labels.shape[0] == all_pooled_frames.shape[0], "Shape mismatch occured"

    unique_labels, unique_idxs = [
        torch.from_numpy(t) for t in np.unique(all_step_labels.detach().cpu().numpy(), return_index=True)]
    unique_step_features = all_step_features[unique_idxs]
    N_steps = all_pooled_frames.shape[0]

    # creating the matrix of targets for the MIL-NCE contrastive objective
    xz_label_mat = torch.zeros([N_steps, unique_labels.shape[0]]).to(all_pooled_frames.device)
    for i in range(all_pooled_frames.shape[0]):
        for j in range(unique_labels.shape[0]):
            xz_label_mat[i, j] = pooled_frames_labels[i] == unique_labels[j]

    all_pooled_frames = F.normalize(all_pooled_frames, p=2, dim=1)
    unique_step_features = F.normalize(unique_step_features, p=2, dim=1)
    return mil_nce(
        all_pooled_frames,
        unique_step_features,
        xz_label_mat,
        gamma=score_only_temperature,
        hard_ratio=1,
    )


def compute_alignment_loss(samples, drop_cost_scale=1.5):
    zx_costs_list = []
    drop_costs_list = []
    for sample in samples:
        zx_costs, drop_costs, _ = compute_all_costs(
            sample,
            10.0,
            0.3,
            distinct_step_occurrences=True,
        )
        drop_costs = drop_costs * float(drop_cost_scale)
        zx_costs_list.append(zx_costs)
        drop_costs_list.append(drop_costs)
    min_costs, _ = batch_dropDTW(
        zx_costs_list,
        drop_costs_list,
        gamma_min=1.0,
        drop_mode="DropDTW",
        contiguous=True,
        softning="prob",
    )
    return sum(cost / len(samples) for cost in min_costs)
