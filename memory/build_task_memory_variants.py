import argparse
import csv
import json
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def feature_value(record, primary, fallback):
    if primary in record:
        return record[primary]
    if fallback in record:
        return record[fallback]
    raise KeyError(primary)


def l2_normalize(array, axis=1, eps=1e-8):
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, eps)


def standardize_columns(array, eps=1e-6):
    array = np.asarray(array, dtype=np.float32)
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    return (array - mean) / np.maximum(std, eps)


def trimmed_visual_prototype(features, trim_fraction):
    """Return a robust mean after removing visual outliers around a spherical center."""
    count = int(features.shape[0])
    if count < 1:
        raise ValueError("Cannot construct a prototype from an empty cluster")
    keep = max(1, int(math.ceil(count * (1.0 - float(trim_fraction)))))
    normalized = F.normalize(features.float(), p=2, dim=1)
    center = F.normalize(normalized.mean(dim=0, keepdim=True), p=2, dim=1)
    similarities = (normalized @ center.transpose(0, 1)).squeeze(1)
    retained = torch.topk(similarities, k=keep, largest=True).indices
    return features[retained].mean(dim=0), retained


def spherical_kmeans(features, k, seed=0, max_iter=100):
    count = features.shape[0]
    if k <= 1 or count <= 1:
        return np.zeros(count, dtype=np.int64)
    k = min(k, count)
    rng = np.random.default_rng(seed)
    centroids = features[rng.choice(count, size=k, replace=False)].copy()
    centroids = l2_normalize(centroids)
    labels = np.full(count, -1, dtype=np.int64)
    for _ in range(max_iter):
        new_labels = np.argmax(features @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = features[labels == cluster_id]
            if len(members) == 0:
                centroids[cluster_id] = features[rng.integers(0, count)]
            else:
                centroids[cluster_id] = members.mean(axis=0)
        centroids = l2_normalize(centroids)
    return labels


def relabel_contiguous(labels):
    mapping = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int64)


def merge_small_clusters(labels, features, min_support):
    labels = relabel_contiguous(labels)
    while True:
        unique, counts = np.unique(labels, return_counts=True)
        small = [int(label) for label, count in zip(unique, counts) if count < min_support]
        if not small or len(unique) == 1:
            return relabel_contiguous(labels)

        source = min(small, key=lambda label: int((labels == label).sum()))
        source_center = l2_normalize(features[labels == source].mean(axis=0, keepdims=True))[0]
        candidates = [int(label) for label in unique if int(label) != source]
        target = max(
            candidates,
            key=lambda label: float(
                source_center
                @ l2_normalize(features[labels == label].mean(axis=0, keepdims=True))[0]
            ),
        )
        labels[labels == source] = target
        labels = relabel_contiguous(labels)


def allocate_budgets(step_counts, total_budget, min_support):
    step_ids = sorted(step_counts)
    capacities = {
        step_id: max(1, int(step_counts[step_id]) // min_support)
        for step_id in step_ids
    }
    budgets = {step_id: 1 for step_id in step_ids}
    target = min(total_budget, sum(capacities.values()))
    while sum(budgets.values()) < target:
        candidates = [step_id for step_id in step_ids if budgets[step_id] < capacities[step_id]]
        if not candidates:
            break
        # Greedy allocation whose equilibrium is approximately proportional
        # to sqrt(number of occurrences), without exceeding support capacity.
        selected = max(
            candidates,
            key=lambda step_id: (
                math.sqrt(step_counts[step_id]) / (budgets[step_id] + 1),
                step_counts[step_id],
                -step_id,
            ),
        )
        budgets[selected] += 1
    return budgets


def add_occurrence_phase(records):
    by_video = defaultdict(list)
    for record_index, record in enumerate(records):
        by_video[(int(record["task_id"]), str(record["video_id"]))].append(
            (record_index, record)
        )

    phases = {}
    for route in by_video.values():
        route = sorted(route, key=lambda item: int(item[1]["step_index"]))
        totals = Counter(int(record["step_id"]) for _, record in route)
        seen = Counter()
        for record_index, record in route:
            step_id = int(record["step_id"])
            seen[step_id] += 1
            rank = seen[step_id]
            total = totals[step_id]
            rank_normalized = 0.0 if total <= 1 else (rank - 1) / (total - 1)
            phases[record_index] = {
                "occurrence_rank": rank,
                "occurrences_in_route": total,
                "occurrence_rank_normalized": rank_normalized,
            }
    return phases


def load_descriptions(path):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return {int(key): str(value) for key, value in payload["steps_to_descriptions"].items()}


def load_held_out_ids(path):
    with open(path, newline="") as handle:
        return {int(row["task_id"]) for row in csv.DictReader(handle)}


def build_step_variants(task_id, step_id, indexed_records, budget, descriptions, args):
    raw_visual = np.stack(
        [
            as_numpy(feature_value(record, "raw_visual_feature", "visual_feature"))
            for _, record in indexed_records
        ]
    )
    temporal = np.stack([as_numpy(record["temporal_feature"]) for _, record in indexed_records])
    phase = np.asarray(
        [record["occurrence_rank_normalized"] for _, record in indexed_records],
        dtype=np.float32,
    )[:, None]

    # Balance modalities before concatenation.  The previous raw
    # ``3 * temporal`` block commonly had a larger norm than the unit visual
    # vector, so variants were grouped mainly by time rather than appearance.
    visual_for_clustering = l2_normalize(raw_visual)
    temporal_for_clustering = standardize_columns(temporal)
    phase_for_clustering = standardize_columns(phase)
    cluster_features = np.concatenate(
        [
            visual_for_clustering,
            args.temporal_weight * temporal_for_clustering,
            args.phase_weight * phase_for_clustering,
        ],
        axis=1,
    ).astype(np.float32)
    cluster_features = l2_normalize(cluster_features)
    labels = spherical_kmeans(
        cluster_features,
        budget,
        seed=args.seed + task_id * 1009 + step_id,
        max_iter=args.max_iter,
    )
    labels = merge_small_clusters(labels, cluster_features, args.min_support)

    nodes = []
    assignments = []
    for variant_id, cluster_id in enumerate(sorted(set(labels.tolist()))):
        member_positions = np.where(labels == cluster_id)[0]
        members = [indexed_records[position] for position in member_positions]
        member_records = [record for _, record in members]
        member_visual = torch.stack(
            [
                feature_value(record, "raw_visual_feature", "visual_feature").float()
                for record in member_records
            ]
        )
        member_text = torch.stack(
            [
                feature_value(record, "raw_text_feature", "text_feature").float()
                for record in member_records
            ]
        )
        member_temporal = torch.stack(
            [record["temporal_feature"].float() for record in member_records]
        )
        raw_visual_proto, retained_visual_indices = trimmed_visual_prototype(
            member_visual, args.prototype_trim_fraction
        )
        retained_visual = member_visual[retained_visual_indices]
        temporal_median = member_temporal.median(dim=0).values
        node = {
            "task_id": task_id,
            "canonical_step_id": step_id,
            "variant_id_within_step": variant_id,
            "step_text": descriptions.get(step_id),
            "raw_text_proto": member_text.mean(dim=0),
            "raw_text_variance": member_text.var(dim=0, unbiased=False),
            "raw_visual_proto": raw_visual_proto,
            "raw_visual_variance": retained_visual.var(dim=0, unbiased=False),
            "visual_prototype_aggregation": "trimmed_spherical_mean",
            "visual_prototype_retained_count": int(retained_visual.shape[0]),
            "temporal_proto": temporal_median,
            "temporal_mad": (member_temporal - temporal_median).abs().median(dim=0).values,
            "occurrence_rank_median": float(
                np.median([record["occurrence_rank"] for record in member_records])
            ),
            "occurrence_rank_normalized_median": float(
                np.median(
                    [record["occurrence_rank_normalized"] for record in member_records]
                )
            ),
            "occurrence_count": len(member_records),
        }
        nodes.append(node)
        for record_index, record in members:
            assignments.append(
                {
                    "record_index": int(record_index),
                    "task_id": task_id,
                    "video_id": str(record["video_id"]),
                    "step_index": int(record["step_index"]),
                    "canonical_step_id": step_id,
                    "variant_id_within_step": variant_id,
                }
            )
    return nodes, assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrences", required=True)
    parser.add_argument("--steps_info", required=True)
    parser.add_argument("--held_out_tasks_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--node_budget", type=int, default=50)
    parser.add_argument("--min_support", type=int, default=3)
    parser.add_argument("--temporal_weight", type=float, default=0.5)
    parser.add_argument("--phase_weight", type=float, default=0.25)
    parser.add_argument("--prototype_trim_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=100)
    args = parser.parse_args()

    if args.temporal_weight < 0.0 or args.phase_weight < 0.0:
        parser.error("clustering modality weights must be non-negative")
    if not 0.0 <= args.prototype_trim_fraction < 1.0:
        parser.error("--prototype_trim_fraction must be in [0, 1)")

    payload = torch.load(args.occurrences, map_location="cpu")
    records = payload["records"]
    descriptions = load_descriptions(args.steps_info)
    held_out_ids = load_held_out_ids(args.held_out_tasks_csv)
    task_ids = {int(record["task_id"]) for record in records}
    overlap = sorted(task_ids & held_out_ids)
    if overlap:
        raise ValueError(f"Held-out tasks leaked into memory construction: {overlap}")

    phases = add_occurrence_phase(records)
    grouped = defaultdict(list)
    for record_index, record in enumerate(records):
        if not bool(record["is_matched"]):
            continue
        enriched = dict(record)
        enriched.update(phases[record_index])
        grouped[(int(record["task_id"]), int(record["step_id"]))].append(
            (record_index, enriched)
        )

    by_task_step = defaultdict(dict)
    for (task_id, step_id), values in grouped.items():
        by_task_step[task_id][step_id] = values

    tasks = {}
    all_assignments = []
    task_summaries = {}
    for task_id in sorted(by_task_step):
        step_groups = by_task_step[task_id]
        step_counts = {step_id: len(values) for step_id, values in step_groups.items()}
        budgets = allocate_budgets(step_counts, args.node_budget, args.min_support)
        task_nodes = []
        task_assignments = []
        for step_id in sorted(step_groups):
            nodes, assignments = build_step_variants(
                task_id,
                step_id,
                step_groups[step_id],
                budgets[step_id],
                descriptions,
                args,
            )
            for node in nodes:
                node["node_id"] = len(task_nodes)
                variant_id = int(node["variant_id_within_step"])
                for assignment in assignments:
                    if int(assignment["variant_id_within_step"]) == variant_id:
                        assignment["node_id"] = int(node["node_id"])
                task_nodes.append(node)
            task_assignments.extend(assignments)

        tasks[task_id] = task_nodes
        all_assignments.extend(task_assignments)
        supports = [int(node["occurrence_count"]) for node in task_nodes]
        task_summaries[str(task_id)] = {
            "nodes": len(task_nodes),
            "canonical_steps": len(step_groups),
            "requested_budget": args.node_budget,
            "allocated_budgets": {str(k): int(v) for k, v in sorted(budgets.items())},
            "min_node_support": min(supports),
            "max_node_support": max(supports),
            "occurrences": sum(supports),
        }

    total_nodes = sum(len(nodes) for nodes in tasks.values())
    supports = [int(node["occurrence_count"]) for nodes in tasks.values() for node in nodes]
    summary = {
        "schema_version": 2,
        "construction": "canonical_step_multimodal_variants",
        "node_budget_per_task": args.node_budget,
        "min_support": args.min_support,
        "temporal_weight": args.temporal_weight,
        "phase_weight": args.phase_weight,
        "clustering_modality_scaling": "visual_l2_temporal_phase_zscore",
        "visual_prototype_aggregation": "trimmed_spherical_mean",
        "prototype_trim_fraction": args.prototype_trim_fraction,
        "num_tasks": len(tasks),
        "total_nodes": total_nodes,
        "total_assignments": len(all_assignments),
        "unmatched_occurrences_skipped": sum(
            not bool(record["is_matched"]) for record in records
        ),
        "held_out_task_overlap": overlap,
        "global_min_node_support": min(supports),
        "global_max_node_support": max(supports),
        "tasks": task_summaries,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "tasks": tasks,
            "assignments": all_assignments,
            "summary": summary,
        },
        args.output,
    )
    with open(args.summary, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Saved {total_nodes} variants for {len(tasks)} tasks to {args.output}")
    print(f"Saved {len(all_assignments)} occurrence-to-node assignments")
    print(
        f"Node support min/max: {summary['global_min_node_support']}/"
        f"{summary['global_max_node_support']}"
    )
    print(f"Held-out overlap: {overlap}")


if __name__ == "__main__":
    main()
