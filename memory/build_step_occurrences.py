import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.data_module import DataModule
from dp.dp_utils import compute_all_costs
from dp.exact_dp import drop_dtw
from models.nets import EmbeddingsMapping


DEFAULT_CKPT = (
    "best_models/drop_dtw_coin/best1_iou_seed40_epoch09.ckpt"
)


def scalar_int(value):
    if torch.is_tensor(value):
        return int(value.detach().cpu().numpy())
    return int(value)


def load_model(args, device):
    model = EmbeddingsMapping(
        d=512,
        learnable_drop=(args.drop_cost == "learn"),
        video_layers=args.video_layers,
        text_layers=args.text_layers,
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["state_dict"]
    state_dict = {
        k[len("model.") :]: v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def infer_assignment(sample, model, args, device):
    frame_features = model.map_video(sample["frame_features"].to(device)).detach()
    step_features = model.map_text(sample["step_features"].to(device)).detach()

    work_sample = dict(sample)
    work_sample["frame_features"] = frame_features.cpu()
    work_sample["step_features"] = step_features.cpu()

    if args.drop_cost == "learn":
        distractor = model.compute_distractors(
            step_features.mean(0).to(device)
        ).detach().cpu()
    else:
        distractor = None

    zx_costs, drop_costs, _ = compute_all_costs(
        work_sample,
        distractor,
        args.gamma,
        drop_cost_type=args.drop_cost,
        keep_percentile=args.keep_percentile,
    )
    assignment = drop_dtw(
        zx_costs.detach().cpu().numpy(),
        drop_costs.detach().cpu().numpy(),
        return_labels=True,
    ) - 1
    return assignment.astype(np.int64), work_sample


def map_sample_features(sample, model, device):
    frame_features = model.map_video(sample["frame_features"].to(device)).detach()
    step_features = model.map_text(sample["step_features"].to(device)).detach()
    work_sample = dict(sample)
    work_sample["frame_features"] = frame_features.cpu()
    work_sample["step_features"] = step_features.cpu()
    return work_sample


def robust_span_pool(
    mapped_frames,
    raw_frames,
    mapped_step,
    top_fraction=0.6,
    temperature=0.1,
):
    """Pool the most step-consistent aligned frames with soft confidence weights."""
    if mapped_frames.ndim != 2 or mapped_frames.shape[0] < 1:
        raise ValueError("robust_span_pool requires at least one aligned frame")
    keep = max(1, int(np.ceil(mapped_frames.shape[0] * float(top_fraction))))
    frame_view = F.normalize(mapped_frames.float(), p=2, dim=1)
    step_view = F.normalize(mapped_step.float().reshape(1, -1), p=2, dim=1)
    scores = (frame_view @ step_view.transpose(0, 1)).squeeze(1)
    selected_scores, selected = torch.topk(scores, k=keep, largest=True)
    weights = torch.softmax(selected_scores / float(temperature), dim=0)
    mapped_visual = (mapped_frames[selected] * weights.unsqueeze(1)).sum(dim=0)
    raw_visual = (raw_frames[selected] * weights.unsqueeze(1)).sum(dim=0)
    return mapped_visual, raw_visual, keep, float(selected_scores.mean())


def make_occurrences(dataset, model, args, device):
    records = []
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": args.ckpt,
        "drop_cost": args.drop_cost,
        "gamma": args.gamma,
        "keep_percentile": args.keep_percentile,
        "schema_version": 2,
        "raw_feature_fields": ["raw_text_feature", "raw_visual_feature"],
        "visual_pooling": "top_fraction_step_similarity_softmax",
        "visual_pool_top_fraction": args.visual_pool_top_fraction,
        "visual_pool_temperature": args.visual_pool_temperature,
        "visual_pool_candidate_frames": 0,
        "visual_pool_selected_frames": 0,
        "visual_pool_confidence_sum": 0.0,
        "num_videos": 0,
        "num_steps": 0,
        "num_matched_steps": 0,
        "num_unmatched_steps": 0,
        "tasks": defaultdict(lambda: {
            "videos": 0,
            "steps": 0,
            "matched_steps": 0,
            "unmatched_steps": 0,
        }),
    }
    task_video_seen = defaultdict(set)
    unmatched_reasons = Counter()

    for sample in tqdm(dataset, desc="Pseudo-labeling"):
        num_steps = scalar_int(sample["num_steps"])
        if num_steps < 1:
            continue

        # Keep encoder-input features in the graph-construction records.  The
        # mapped features below belong to the checkpoint used for initial
        # pseudo-labeling and may become stale when the alignment model is
        # fine-tuned.  Raw prototypes can instead be passed through the current
        # video/text mappings when task-memory nodes are formed.
        raw_frame_features = sample["frame_features"].detach().cpu()
        raw_step_features = sample["step_features"].detach().cpu()

        if args.span_source == "pseudo":
            assignment, mapped_sample = infer_assignment(sample, model, args, device)
        else:
            assignment = None
            mapped_sample = map_sample_features(sample, model, device)
        num_frames = scalar_int(mapped_sample["num_frames"])
        task_id = scalar_int(mapped_sample["cls"])
        task_name = mapped_sample.get("cls_name", str(task_id))
        video_id = mapped_sample.get("name", "")

        summary["num_videos"] += 1
        task_video_seen[task_id].add(video_id)

        for step_index in range(num_steps):
            step_id = scalar_int(mapped_sample["step_ids"][step_index])
            if args.span_source == "pseudo":
                matched_frames = np.where(assignment == step_index)[0]
                is_matched = matched_frames.size > 0
                if is_matched:
                    pseudo_start = int(matched_frames[0])
                    pseudo_end = int(matched_frames[-1])
                    aligned_indices = torch.from_numpy(matched_frames).long()
                else:
                    pseudo_start = -1
                    pseudo_end = -1
                    aligned_indices = None
            else:
                pseudo_start = scalar_int(mapped_sample["step_starts"][step_index])
                pseudo_end = scalar_int(mapped_sample["step_ends"][step_index])
                is_matched = pseudo_start >= 0 and pseudo_end >= pseudo_start
                aligned_indices = None

            if is_matched:
                pseudo_start = max(0, min(pseudo_start, num_frames - 1))
                pseudo_end = max(pseudo_start, min(pseudo_end, num_frames - 1))
                if aligned_indices is None:
                    aligned_indices = torch.arange(
                        pseudo_start, pseudo_end + 1, dtype=torch.long
                    )
                aligned_indices = aligned_indices.clamp(0, num_frames - 1)
                span_features = mapped_sample["frame_features"][aligned_indices]
                raw_span_features = raw_frame_features[aligned_indices]
                (
                    visual_feature,
                    raw_visual_feature,
                    selected_frame_count,
                    visual_pool_confidence,
                ) = robust_span_pool(
                    span_features,
                    raw_span_features,
                    mapped_sample["step_features"][step_index],
                    top_fraction=args.visual_pool_top_fraction,
                    temperature=args.visual_pool_temperature,
                )
                duration = max(pseudo_end - pseudo_start + 1, 1)
                temporal_feature = torch.tensor(
                    [
                        pseudo_start / max(num_frames, 1),
                        pseudo_end / max(num_frames, 1),
                        duration / max(num_frames, 1),
                    ],
                    dtype=torch.float32,
                )
                summary["num_matched_steps"] += 1
                summary["tasks"][task_id]["matched_steps"] += 1
                summary["visual_pool_candidate_frames"] += int(span_features.shape[0])
                summary["visual_pool_selected_frames"] += int(selected_frame_count)
                summary["visual_pool_confidence_sum"] += visual_pool_confidence
            else:
                pseudo_start = -1
                pseudo_end = -1
                visual_feature = torch.zeros_like(mapped_sample["frame_features"][0])
                raw_visual_feature = torch.zeros_like(raw_frame_features[0])
                temporal_feature = torch.tensor([-1.0, -1.0, -1.0])
                selected_frame_count = 0
                visual_pool_confidence = 0.0
                summary["num_unmatched_steps"] += 1
                summary["tasks"][task_id]["unmatched_steps"] += 1
                unmatched_reasons["no_assigned_clip"] += 1

            step_text = None
            if "steps" in mapped_sample:
                try:
                    step_text = mapped_sample["steps"][step_index]
                except Exception:
                    step_text = None

            records.append(
                {
                    "task_id": task_id,
                    "task_name": task_name,
                    "video_id": video_id,
                    "step_index": step_index,
                    "step_id": step_id,
                    "step_text": step_text,
                    "text_feature": mapped_sample["step_features"][step_index].cpu(),
                    "raw_text_feature": raw_step_features[step_index].cpu(),
                    "visual_feature": visual_feature.cpu(),
                    "raw_visual_feature": raw_visual_feature.cpu(),
                    "temporal_feature": temporal_feature.cpu(),
                    "pseudo_start": pseudo_start,
                    "pseudo_end": pseudo_end,
                    "num_frames": num_frames,
                    "is_matched": is_matched,
                    "visual_pool_selected_frames": selected_frame_count,
                    "visual_pool_confidence": visual_pool_confidence,
                }
            )
            summary["num_steps"] += 1
            summary["tasks"][task_id]["steps"] += 1

    for task_id, videos in task_video_seen.items():
        summary["tasks"][task_id]["videos"] = len(videos)
    summary["tasks"] = {str(k): v for k, v in summary["tasks"].items()}
    summary["unmatched_reasons"] = dict(unmatched_reasons)
    summary["visual_pool_confidence_mean"] = (
        summary["visual_pool_confidence_sum"] / max(summary["num_matched_steps"], 1)
    )
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="COIN", choices=["COIN"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", default="tam_outputs/COIN/task_memory")
    parser.add_argument("--span_source", default="pseudo", choices=["pseudo", "gt"])
    parser.add_argument("--drop_cost", default="learn", choices=["learn", "logit"])
    parser.add_argument("--gamma", type=float, default=30.0)
    parser.add_argument("--keep_percentile", type=float, default=0.3)
    parser.add_argument("--visual_pool_top_fraction", type=float, default=0.6)
    parser.add_argument("--visual_pool_temperature", type=float, default=0.1)
    parser.add_argument("--video_layers", type=int, default=2)
    parser.add_argument("--text_layers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cls", type=int, default=1)
    args = parser.parse_args()

    if not 0.0 < args.visual_pool_top_fraction <= 1.0:
        parser.error("--visual_pool_top_fraction must be in (0, 1]")
    if args.visual_pool_temperature <= 0.0:
        parser.error("--visual_pool_temperature must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args, device)

    data = DataModule(args.dataset, args.n_cls, args.batch_size)
    dataset = getattr(data, f"{args.split}_dataset")

    records, summary = make_occurrences(dataset, model, args, device)

    pt_path = os.path.join(
        args.output_dir, f"coin_{args.split}_step_occurrences_{args.span_source}.pt"
    )
    json_path = os.path.join(
        args.output_dir, f"coin_{args.split}_step_occurrences_{args.span_source}_summary.json"
    )
    torch.save({"records": records, "summary": summary}, pt_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(records)} step occurrence records to {pt_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
