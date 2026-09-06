import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def median_mad(values):
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float32)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, mad


def semantic_edges(nodes, topk):
    by_step = defaultdict(list)
    for node in nodes:
        by_step[int(node["canonical_step_id"])].append(int(node["node_id"]))

    step_ids = sorted(by_step)
    anchors = []
    for step_id in step_ids:
        step_nodes = by_step[step_id]
        anchors.append(
            torch.stack([nodes[node_id]["raw_text_proto"].float() for node_id in step_nodes]).mean(0)
        )
    anchors = F.normalize(torch.stack(anchors), p=2, dim=1)
    similarity = anchors @ anchors.T

    canonical_pairs = set()
    for source_index, source_step in enumerate(step_ids):
        candidates = [index for index in range(len(step_ids)) if index != source_index]
        candidates.sort(key=lambda index: float(similarity[source_index, index]), reverse=True)
        if topk > 0:
            candidates = candidates[:topk]
        for target_index in candidates:
            target_step = step_ids[target_index]
            canonical_pairs.add(tuple(sorted((source_step, target_step))))

    representatives = {
        step_id: max(
            by_step[step_id],
            key=lambda node_id: (int(nodes[node_id]["occurrence_count"]), -node_id),
        )
        for step_id in step_ids
    }
    step_to_index = {step_id: index for index, step_id in enumerate(step_ids)}
    edges = {}

    # Variants of the same semantic step form a sparse star around the most
    # supported prototype.  This keeps variant identity explicit without
    # allowing same-step clones to occupy all semantic neighbors.
    for step_id, node_ids in by_step.items():
        representative = representatives[step_id]
        for node_id in node_ids:
            if node_id == representative:
                continue
            for source, target in ((node_id, representative), (representative, node_id)):
                edge = edges.setdefault((source, target), {})
                edge["semantic_similarity"] = 1.0
                edge["variant_relation"] = True

    # A canonical semantic relation is routed through representative variants
    # instead of forming a dense Cartesian product between all variants.
    for source_step, target_step in sorted(canonical_pairs):
        source_rep = representatives[source_step]
        target_rep = representatives[target_step]
        score = float(
            similarity[step_to_index[source_step], step_to_index[target_step]]
        )
        for source_node in by_step[source_step]:
            for source, target in ((source_node, target_rep), (target_rep, source_node)):
                edge = edges.setdefault((source, target), {})
                edge["semantic_similarity"] = score
                edge["variant_relation"] = False
        for target_node in by_step[target_step]:
            for source, target in ((target_node, source_rep), (source_rep, target_node)):
                edge = edges.setdefault((source, target), {})
                edge["semantic_similarity"] = score
                edge["variant_relation"] = False
    return edges


def build_routes(records, assignments):
    assignment_by_record = {
        int(assignment["record_index"]): int(assignment["node_id"])
        for assignment in assignments
    }
    routes = defaultdict(list)
    for record_index, record in enumerate(records):
        if record_index not in assignment_by_record:
            continue
        routes[(int(record["task_id"]), str(record["video_id"]))].append(
            {
                "node_id": assignment_by_record[record_index],
                "canonical_step_id": int(record["step_id"]),
                "step_index": int(record["step_index"]),
                "pseudo_start": int(record["pseudo_start"]),
                "pseudo_end": int(record["pseudo_end"]),
                "num_frames": int(record["num_frames"]),
            }
        )
    for key in routes:
        routes[key].sort(
            key=lambda item: (item["pseudo_start"], item["pseudo_end"], item["step_index"])
        )
    return routes


def add_temporal_statistics(edges, task_routes, min_temporal_support):
    occurrence_counts = Counter()
    adjacent_counts = Counter()
    precedence_counts = Counter()
    gaps = defaultdict(list)

    for route in task_routes:
        for item in route:
            occurrence_counts[item["node_id"]] += 1
        for source, target in zip(route[:-1], route[1:]):
            adjacent_counts[(source["node_id"], target["node_id"])] += 1
        for source_index in range(len(route)):
            for target_index in range(source_index + 1, len(route)):
                source = route[source_index]
                target = route[target_index]
                pair = (source["node_id"], target["node_id"])
                precedence_counts[pair] += 1
                gaps[pair].append(
                    (target["pseudo_start"] - source["pseudo_end"])
                    / max(source["num_frames"], 1)
                )

    temporal_pairs = {
        pair
        for pair in (set(precedence_counts) | set(adjacent_counts))
        if precedence_counts[pair] >= min_temporal_support
    }
    outgoing_adjacent = Counter()
    for (source, target), count in adjacent_counts.items():
        if (source, target) in temporal_pairs:
            outgoing_adjacent[source] += count
    for source, target in temporal_pairs:
        forward = int(precedence_counts[(source, target)])
        reverse = int(precedence_counts[(target, source)])
        denominator = forward + reverse
        if source == target:
            # A repeated occurrence assigned to the same variant is a valid
            # self transition; it has no meaningful reverse ordering.
            precedence_probability = 1.0
        else:
            precedence_probability = forward / denominator if denominator else 0.0
        transition_count = int(adjacent_counts[(source, target)])
        transition_probability = (
            transition_count / outgoing_adjacent[source]
            if outgoing_adjacent[source]
            else 0.0
        )
        gap_median, gap_mad = median_mad(gaps[(source, target)])
        edge = edges.setdefault((source, target), {})
        edge.update(
            {
                "transition_count": transition_count,
                "transition_probability": float(transition_probability),
                "precedence_count": forward,
                "reverse_precedence_count": reverse,
                "precedence_probability": float(precedence_probability),
                "normalized_gap_median": gap_median,
                "normalized_gap_mad": gap_mad,
            }
        )
    return occurrence_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--occurrences", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--semantic_topk_steps", type=int, default=3)
    parser.add_argument("--min_temporal_support", type=int, default=2)
    args = parser.parse_args()

    occurrence_payload = torch.load(args.occurrences, map_location="cpu")
    records = occurrence_payload["records"]
    variant_payload = torch.load(args.variants, map_location="cpu")
    tasks = {int(task_id): nodes for task_id, nodes in variant_payload["tasks"].items()}
    assignments = variant_payload["assignments"]
    routes = build_routes(records, assignments)

    routes_by_task = defaultdict(list)
    for (task_id, _), route in routes.items():
        if route:
            routes_by_task[task_id].append(route)

    graphs = {}
    task_summaries = {}
    for task_id in sorted(tasks):
        nodes = tasks[task_id]
        edges = semantic_edges(nodes, args.semantic_topk_steps)
        occurrence_counts = add_temporal_statistics(
            edges,
            routes_by_task.get(task_id, []),
            args.min_temporal_support,
        )

        edge_records = []
        semantic_count = temporal_count = same_step_temporal_count = self_temporal_count = 0
        variant_relation_count = cross_step_semantic_count = 0
        for (source, target), attributes in sorted(edges.items()):
            source_step = int(nodes[source]["canonical_step_id"])
            target_step = int(nodes[target]["canonical_step_id"])
            has_semantic = "semantic_similarity" in attributes
            has_temporal = "precedence_count" in attributes
            semantic_count += int(has_semantic)
            variant_relation = bool(attributes.get("variant_relation", False))
            variant_relation_count += int(has_semantic and variant_relation)
            cross_step_semantic_count += int(has_semantic and source_step != target_step)
            temporal_count += int(has_temporal)
            same_step_temporal_count += int(has_temporal and source_step == target_step)
            self_temporal_count += int(has_temporal and source == target)
            edge_records.append(
                {
                    "src": int(source),
                    "dst": int(target),
                    "src_step_id": source_step,
                    "dst_step_id": target_step,
                    "same_canonical_step": source_step == target_step,
                    "semantic": {
                        "similarity": float(attributes.get("semantic_similarity", 0.0)),
                        "has_semantic": bool(has_semantic),
                        "variant_relation": variant_relation,
                    },
                    "temporal": {
                        "transition_count": int(attributes.get("transition_count", 0)),
                        "transition_probability": float(
                            attributes.get("transition_probability", 0.0)
                        ),
                        "precedence_count": int(attributes.get("precedence_count", 0)),
                        "reverse_precedence_count": int(
                            attributes.get("reverse_precedence_count", 0)
                        ),
                        "precedence_probability": float(
                            attributes.get("precedence_probability", 0.0)
                        ),
                        "normalized_gap_median": float(
                            attributes.get("normalized_gap_median", 0.0)
                        ),
                        "normalized_gap_mad": float(
                            attributes.get("normalized_gap_mad", 0.0)
                        ),
                        "has_temporal": bool(has_temporal),
                    },
                }
            )

        graphs[task_id] = {"nodes": nodes, "edges": edge_records}
        task_summaries[str(task_id)] = {
            "nodes": len(nodes),
            "routes": len(routes_by_task.get(task_id, [])),
            "edges": len(edge_records),
            "semantic_edges": semantic_count,
            "variant_relation_edges": variant_relation_count,
            "cross_step_semantic_edges": cross_step_semantic_count,
            "temporal_edges": temporal_count,
            "same_step_temporal_edges": same_step_temporal_count,
            "self_temporal_edges": self_temporal_count,
            "nodes_with_occurrences": len(occurrence_counts),
        }

    summary = {
        "schema_version": 2,
        "construction": "multimodal_variant_graph",
        "source_variant_file": os.path.abspath(args.variants),
        "source_occurrence_file": os.path.abspath(args.occurrences),
        "semantic_topk_steps": args.semantic_topk_steps,
        "min_temporal_support": args.min_temporal_support,
        "num_tasks": len(graphs),
        "total_nodes": sum(len(graph["nodes"]) for graph in graphs.values()),
        "total_edges": sum(len(graph["edges"]) for graph in graphs.values()),
        "semantic_edges": sum(item["semantic_edges"] for item in task_summaries.values()),
        "variant_relation_edges": sum(
            item["variant_relation_edges"] for item in task_summaries.values()
        ),
        "cross_step_semantic_edges": sum(
            item["cross_step_semantic_edges"] for item in task_summaries.values()
        ),
        "temporal_edges": sum(item["temporal_edges"] for item in task_summaries.values()),
        "same_step_temporal_edges": sum(
            item["same_step_temporal_edges"] for item in task_summaries.values()
        ),
        "self_temporal_edges": sum(
            item["self_temporal_edges"] for item in task_summaries.values()
        ),
        "tasks": task_summaries,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    torch.save({"schema_version": 2, "graphs": graphs, "summary": summary}, args.output)
    with open(args.summary, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Saved {summary['num_tasks']} graphs with {summary['total_nodes']} nodes")
    print(
        f"Edges total/semantic/temporal: {summary['total_edges']}/"
        f"{summary['semantic_edges']}/{summary['temporal_edges']}"
    )
    print(
        f"Repeated-step temporal edges/self temporal edges: "
        f"{summary['same_step_temporal_edges']}/{summary['self_temporal_edges']}"
    )
    print(f"Saved graph to {args.output}")


if __name__ == "__main__":
    main()
