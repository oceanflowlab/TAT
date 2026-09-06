import torch
import torch.nn.functional as F
from torch import log, exp
import numpy as np

from dp.soft_dp import batch_dropDTW, batch_NW
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


def compute_clust_loss(samples, distractors, l2_normalize=False, frame_gamma=10,
                      xz_gamma=10, xz_hard_ratio=0.3, all_classes_distinct=False,
                      bg_scope='global'):
    # aggregating videos with attentino for their steps, i.e. done per each step
    all_pooled_frames, pooled_frames_labels = [], []
    all_step_features, all_step_labels = [], []
    global_step_id_count = 0
    score_only_temperature = None
    for i, sample in enumerate(samples):
        step_features, frame_features = sample['step_features'], sample['frame_features']
        pairwise_scores = sample.get('pairwise_scores')
        if pairwise_scores is not None:
            sample_temperature = float(sample['pairwise_score_temperature'])
            if score_only_temperature is None:
                score_only_temperature = sample_temperature
            elif abs(score_only_temperature - sample_temperature) > 1e-12:
                raise ValueError("Score-only samples must share one temperature")

        # Used for YouCook2, where text descriptions are unique for each step
        if all_classes_distinct:
            n_samples = sample['step_ids'].shape[0]
            step_ids = torch.arange(global_step_id_count,
                                    global_step_id_count + n_samples)
            global_step_id_count += n_samples
        else:
            step_ids = sample['step_ids']

        if distractors is not None:
            bg_step_id = torch.tensor([99999]).to(step_ids.dtype).to(step_ids.device)
            if bg_scope == 'class':
                bg_step_id = bg_step_id + sample['cls']
            if bg_scope == 'video':
                global_step_id_count += 1
                bg_step_id = bg_step_id + global_step_id_count
                
            step_ids = torch.cat([step_ids, bg_step_id])
            step_features = torch.cat([step_features, distractors[i][None, :]])
            if pairwise_scores is not None:
                distractor_scores = (
                    F.normalize(distractors[i], p=2, dim=0)[None, :]
                    @ F.normalize(frame_features, p=2, dim=1).T
                ) / sample_temperature
                pairwise_scores = torch.cat(
                    [pairwise_scores, distractor_scores], dim=0
                )

        if l2_normalize:
            step_features = F.normalize(step_features, p=2, dim=1)
            frame_features = F.normalize(frame_features, p=2, dim=1)

        unique_step_labels, unique_idxs = [
            torch.from_numpy(t) for t in np.unique(step_ids.detach().cpu().numpy(), return_index=True)]
        unique_step_features = step_features[unique_idxs]  # size [K, d]
        if pairwise_scores is not None:
            # Use the exact P_u used by guided Drop-DTW for frame pooling.
            sim = pairwise_scores[unique_idxs]
            frame_weights = F.softmax(sim, dim=1)
        else:
            sim = unique_step_features @ frame_features.T
            frame_weights = F.softmax(sim / frame_gamma, dim=1)
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

    # reinforcing existing alignment with step descriptors
    if score_only_temperature is not None:
        # The cross-video comparison uses the same score definition,
        # cosine / tau, without replacing the stored decoder embeddings.
        all_pooled_frames = F.normalize(all_pooled_frames, p=2, dim=1)
        unique_step_features = F.normalize(unique_step_features, p=2, dim=1)
        clustering_gamma = score_only_temperature
    else:
        clustering_gamma = xz_gamma
    xz_loss = mil_nce(all_pooled_frames, unique_step_features, xz_label_mat,
                      gamma=clustering_gamma, hard_ratio=xz_hard_ratio)
    return xz_loss


def compute_alignment_loss(samples, distractors, l2_normalize=False, drop_cost_type='max',
                           dp_algo='DropDTW', keep_percentile=1, contiguous=True, softning='prob',
                           gamma_xz=10, gamma_min=1, aggregate_loss=True,
                           distinct_step_occurrences=False, drop_cost_scale=1.0):
    gamma_xz = 0.1 if l2_normalize else gamma_xz
    gamma_min = 0.1 if l2_normalize else gamma_min

    # do pre-processing
    zx_costs_list = []
    drop_costs_list = []
    for i, sample in enumerate(samples):
        distractor = None if distractors is None else distractors[i]
        zx_costs, drop_costs, _ = compute_all_costs(sample, distractor, gamma_xz, drop_cost_type,
                                                    keep_percentile, l2_normalize,
                                                    distinct_step_occurrences=distinct_step_occurrences)
        drop_costs = drop_costs * float(drop_cost_scale)
        zx_costs_list.append(zx_costs)
        drop_costs_list.append(drop_costs)

    if dp_algo == 'NW':
        min_costs, _ = batch_NW(zx_costs_list, drop_costs_list, gamma_min=gamma_min, softning=softning)
    else:
        min_costs, _ = batch_dropDTW(zx_costs_list, drop_costs_list, gamma_min=gamma_min,
                                     drop_mode=dp_algo, contiguous=contiguous, softning=softning)
    dtw_losses = [c / len(samples) for c in min_costs]
    if aggregate_loss:
        return sum(dtw_losses)
    else:
        return dtw_losses


def compute_span_recall_loss(samples, gamma_xz=10, topk=3, margin=0.0, aggregate_loss=True):
    """Lightweight training-only span supervision.

    For each annotated step occurrence, encourage at least a few clips inside
    its GT span to score higher than the same step's clips outside the span.
    This uses GT only during training; inference/decode remains GT-free.
    """
    losses = []
    for sample in samples:
        step_features = sample['step_features']
        frame_features = sample['frame_features']
        pairwise_scores = sample.get('pairwise_scores')
        if pairwise_scores is not None:
            sim = pairwise_scores
        else:
            sim = (step_features @ frame_features.T) / gamma_xz
        num_frames = int(frame_features.shape[0])
        num_steps = int(sample['num_steps'])
        sample_losses = []
        for step_idx in range(num_steps):
            start = int(sample['step_starts'][step_idx].detach().cpu().item())
            end = int(sample['step_ends'][step_idx].detach().cpu().item())
            start = max(0, min(start, num_frames - 1))
            end = max(start, min(end, num_frames - 1))
            pos = sim[step_idx, start:end + 1]
            if pos.numel() == 0:
                continue
            k = max(1, min(int(topk), int(pos.numel())))
            pos_score = torch.topk(pos, k=k).values.mean()
            if end - start + 1 >= num_frames:
                # Degenerate all-video span: just make the span confidently
                # visible for this step.
                sample_losses.append(F.softplus(-pos_score))
                continue
            outside_parts = []
            if start > 0:
                outside_parts.append(sim[step_idx, :start])
            if end + 1 < num_frames:
                outside_parts.append(sim[step_idx, end + 1:])
            neg = torch.cat(outside_parts, dim=0)
            neg_k = max(1, min(k, int(neg.numel())))
            neg_score = torch.topk(neg, k=neg_k).values.mean()
            sample_losses.append(F.softplus(neg_score + float(margin) - pos_score))
        if sample_losses:
            losses.append(torch.stack(sample_losses).mean())
    if not losses:
        reference = samples[0]['frame_features'] if samples else torch.tensor(0.0)
        zero = reference.new_zeros(())
        return zero if aggregate_loss else []
    if aggregate_loss:
        return torch.stack(losses).mean()
    return losses
