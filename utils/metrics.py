import os
import numpy as np
from tqdm import tqdm
from dp.exact_dp import crosstask_dp


FEATURE_STRIDE = int(os.environ.get("DROPD_TW_FEATURE_STRIDE", "32"))
CLIP_SECONDS = FEATURE_STRIDE / 10


def framewise_accuracy(frame_assignment, sample, use_unlabeled=False):
    """ calculate framewise accuracy as done in COIN """
    # convert start and end times into clip-level labels
    num_steps = sample['num_steps'].numpy().squeeze()
    num_frames = sample['num_frames'].numpy().squeeze()
    # non-step frames/clips are assigned label = -1
    gt_assignment = -np.ones(num_frames, dtype=np.int32)
    # convert start and end times to clip/frame -wise labels
    for s in np.arange(num_steps):
        st_ed = np.arange(sample['step_starts'][s], sample['step_ends'][s]+1)
        gt_assignment[st_ed] = s 
    
    # to discount unlabeled frames in gt
    if not use_unlabeled:
        unlabled = np.count_nonzero(gt_assignment == -1)
        num_frames = num_frames - unlabled
        fa = np.logical_and(frame_assignment == gt_assignment, gt_assignment != -1).sum()
    else:
        fa = np.count_nonzero((frame_assignment == gt_assignment))
    # framewise accuracy
    fa = fa / num_frames if num_frames != 0 else 0
    return fa


def IoU(frame_assignment, sample):
    """ calculate framewise accuracy as done in COIN """
    # convert start and end times into clip-level labels
    num_steps = sample['num_steps'].numpy()
    num_frames = sample['num_frames'].numpy()
    # non-step frames/clips are assigned label = -1
    gt_assignment = -np.ones((num_frames,), dtype=np.int32)
    # convert start and end times to clip/frame -wise labels
    intersection, union = 0, 0
    for s in range(num_steps):
        st_ed = np.arange(sample['step_starts'][s], sample['step_ends'][s] + 1)
        gt_assignment[st_ed] = s
        intersection += np.logical_and(gt_assignment == s, frame_assignment == s).sum()
        union += np.logical_or(gt_assignment == s, frame_assignment == s).sum()
    return intersection / union


def recall_crosstask(val_loader, model):
    "Recall as defined in CrossTask"
    total_steps = 0
    detected_steps_clips = 0
    detected_steps_seconds = 0
    for i, sample in enumerate(tqdm(val_loader)):
        if sample['num_steps'] < 1:
            continue

        device = "cuda" if (model is not None and next(model.parameters()).is_cuda) else "cpu"
        if model is not None:
            frame_features = model.map_video(sample['frame_features'].to(device)).detach().cpu()
            step_features = model.map_text(sample['step_features'].to(device)).detach().cpu()
        else:
            frame_features = sample['frame_features'].cpu()
            step_features = sample['step_features'].cpu()
        text_clip_similarity = (step_features @ frame_features.T).detach().cpu().numpy() 

        optimal_assignment = crosstask_dp(-text_clip_similarity.T).argmax(0)
        for si in range(sample['num_steps']):
            # eval on seconds
            start, end = sample['step_starts_sec'][si], sample['step_ends_sec'][si]
            infer_t = (optimal_assignment[si] + 0.5) * CLIP_SECONDS
            detected_steps_seconds += int(infer_t >= start and infer_t <= end)

            # eval on clips level
            start, end = sample['step_starts'][si], sample['step_ends'][si]
            infer_t = optimal_assignment[si]
            detected_steps_clips += int(infer_t >= start and infer_t <= end)

            total_steps += 1
            
    sec_recall = 100 * detected_steps_seconds / total_steps
    return sec_recall
