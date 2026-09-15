import torch
import torch.nn.functional as F
from torch import nn


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional route/frame association from Eqs. (21-22)."""

    def __init__(self, d=512, nhead=8, dropout=0.1):
        super().__init__()
        self.route_to_frame = nn.MultiheadAttention(
            d, nhead, dropout=dropout, batch_first=True
        )
        self.frame_to_route = nn.MultiheadAttention(
            d, nhead, dropout=dropout, batch_first=True
        )
        self.route_dropout = nn.Dropout(dropout)
        self.frame_dropout = nn.Dropout(dropout)
        self.route_norm = nn.LayerNorm(d)
        self.frame_query_norm = nn.LayerNorm(d)
        self.route_key_norm = nn.LayerNorm(d)
        self.route_value_norm = nn.LayerNorm(d)
        self.register_buffer("route_residual_scale", torch.tensor(0.1))
        self.register_buffer("frame_residual_scale", torch.tensor(0.1))

    def forward(self, route, frames):
        # Eqs. (21-22) specify the two cross-attention directions.  The paper
        # also states that its Transformer follows [40], so retain the
        # standard residual + dropout + LayerNorm wrapper around each
        # attention sublayer.  This preserves input discrimination instead of
        # replacing every feature by a weighted average of the other stream.
        route_update, _ = self.route_to_frame(
            query=route,
            key=frames,
            value=frames,
            need_weights=False,
        )
        frame_update, _ = self.frame_to_route(
            query=self.frame_query_norm(frames),
            key=self.route_key_norm(route),
            value=self.route_value_norm(route),
            need_weights=False,
        )
        route_candidate = self.route_norm(
            route + self.route_dropout(route_update)
        )
        route_scale = self.route_residual_scale
        frame_scale = self.frame_residual_scale
        enhanced_route = route + route_scale * (route_candidate - route)
        dropped_frame_update = self.frame_dropout(frame_update)
        enhanced_frames = frames + frame_scale * dropped_frame_update
        return enhanced_route, enhanced_frames


class ResidualVisualProjection(nn.Module):
    """Nonlinear visual adapter that preserves the Drop-DTW input space."""

    def __init__(self, d=512, residual_scale=0.1):
        super().__init__()
        self.out_features = int(d)
        self.residual_scale = float(residual_scale)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

    def reset_parameters(self):
        self.norm.reset_parameters()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, features):
        return features + self.residual_scale * self.mlp(self.norm(features))


class TemporalProjection(nn.Module):
    """Nonlinear projection for relative start/end/duration statistics."""

    def __init__(self, time_dim=3, hidden_dim=128, d=512):
        super().__init__()
        self.out_features = int(d)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d),
            nn.LayerNorm(d),
        )

    def reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                module.reset_parameters()

    def forward(self, temporal):
        return self.mlp(temporal)


class LightCrossModalDecoder(nn.Module):
    def __init__(self, d=512, nhead=8, num_layers=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalCrossAttention(d=d, nhead=nhead, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, route, frames):
        route = route.unsqueeze(0)
        frames = frames.unsqueeze(0)
        for layer in self.layers:
            route, frames = layer(route, frames)
        return route.squeeze(0), frames.squeeze(0)


class TaskMemory(nn.Module):
    """Multimodal task memory whose semantic anchors use the current encoder.

    Graph files contain encoder-input text/visual prototypes.  The caller
    supplies the current text/video mappings so memory nodes cannot become
    stranded in the feature space of the pseudo-labeling checkpoint.
    """

    @staticmethod
    def _node_prototype(node, modality):
        key = modality + "_proto"
        if key in node:
            return node[key]
        legacy_key = "".join(("r", "aw_", key))
        if legacy_key in node:
            return node[legacy_key]
        raise KeyError(key)

    def __init__(self, graph_path, assignments_path, d=512):
        super().__init__()
        payload = torch.load(graph_path, map_location="cpu")
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("TaskMemory requires a schema-v2 graph")
        graphs = {int(task_id): graph for task_id, graph in payload["graphs"].items()}
        self.task_ids = sorted(graphs)
        self.task_to_index = {task_id: index for index, task_id in enumerate(self.task_ids)}
        self.assignment_to_node = self._load_assignments(assignments_path)
        self.visual_init_weight = 0.05
        self.time_init_weight = 0.05
        self.graph_update_max_ratio = 0.75
        self.retrieval_temperature = 1.0
        self.guided_score_temperature = 0.02
        self.retriever_local_candidate_topn = 10
        self.retriever_local_alignment_temperature = 0.1
        self.retriever_local_position_sigma = 0.35
        self.retriever_local_visual_weight = 0.08
        self.retriever_local_temporal_weight = 0.02
        for task_index, task_id in enumerate(self.task_ids):
            graph = graphs[task_id]
            nodes = sorted(graph["nodes"], key=lambda node: int(node["node_id"]))
            source_text = torch.stack(
                [self._node_prototype(node, "text").float() for node in nodes]
            )
            source_visual = torch.stack(
                [self._node_prototype(node, "visual").float() for node in nodes]
            )
            temporal = torch.stack([node["temporal_proto"].float() for node in nodes])
            step_ids = torch.tensor(
                [int(node["canonical_step_id"]) for node in nodes], dtype=torch.long
            )
            support = torch.tensor(
                [int(node["occurrence_count"]) for node in nodes], dtype=torch.float32
            )
            rank_prior = torch.tensor(
                [float(node["occurrence_rank_normalized_median"]) for node in nodes],
                dtype=torch.float32,
            )
            self.register_buffer(f"source_text_{task_index}", source_text)
            self.register_buffer(f"source_visual_{task_index}", source_visual)
            self.register_parameter(f"temporal_{task_index}", nn.Parameter(temporal))
            self.register_buffer(f"step_ids_{task_index}", step_ids)
            self.register_buffer(f"support_{task_index}", support)
            self.register_buffer(f"rank_prior_{task_index}", rank_prior)

            edge_index = []
            edge_attributes = []
            for edge in graph["edges"]:
                source = int(edge["src"])
                target = int(edge["dst"])
                semantic = edge["semantic"]
                temporal_edge = edge["temporal"]
                edge_index.append([source, target])
                edge_attributes.append(
                    [
                        float(semantic["similarity"]),
                        float(bool(semantic.get("variant_relation", False))),
                        float(temporal_edge["transition_probability"]),
                        float(temporal_edge["precedence_probability"]),
                        float(temporal_edge["normalized_gap_median"]),
                        float(temporal_edge["normalized_gap_mad"]),
                        float(bool(semantic["has_semantic"])),
                        float(bool(temporal_edge["has_temporal"])),
                        float(bool(edge["same_canonical_step"])),
                        float(source == target),
                    ]
                )
            self.register_buffer(
                f"edge_index_{task_index}", torch.tensor(edge_index, dtype=torch.long)
            )
            edge_attributes = torch.tensor(edge_attributes, dtype=torch.float32)
            self.register_buffer(f"edge_attr_{task_index}", edge_attributes)
            self.register_parameter(
                f"edge_temporal_delta_{task_index}",
                nn.Parameter(torch.zeros((edge_attributes.shape[0], 4))),
            )

        self.semantic_proj = nn.Linear(d, d)
        self.visual_proj = ResidualVisualProjection(d=d, residual_scale=0.1)
        self.time_proj = TemporalProjection(time_dim=3, hidden_dim=128, d=d)
        # Eq. (9): modality-specific projections are concatenated and then
        # projected into the d-dimensional node space.
        self.node_fuse = nn.Linear(3 * d, d)
        self.edge_proj = nn.Sequential(
            nn.Linear(10, d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.edge_message = nn.Linear(d, d)
        self.temporal_ranker = nn.Linear(d, 1, bias=False)
        self.decoder = LightCrossModalDecoder(d=d, nhead=8, num_layers=1, dropout=0.1)
        self.reset_parameters()

    @staticmethod
    def _load_assignments(path):
        if not path:
            return {}
        payload = torch.load(path, map_location="cpu")
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("TaskMemory requires schema-v2 assignments")
        return {
            (
                int(item["task_id"]),
                str(item["video_id"]),
                int(item["step_index"]),
            ): int(item["node_id"])
            for item in payload["assignments"]
        }

    def reset_parameters(self):
        nn.init.eye_(self.semantic_proj.weight)
        nn.init.zeros_(self.semantic_proj.bias)
        self.visual_proj.reset_parameters()
        self.time_proj.reset_parameters()
        nn.init.zeros_(self.node_fuse.weight)
        nn.init.zeros_(self.node_fuse.bias)
        with torch.no_grad():
            d = self.semantic_proj.out_features
            eye = torch.eye(d, device=self.node_fuse.weight.device)
            self.node_fuse.weight[:, :d].copy_(eye)
            self.node_fuse.weight[:, d : 2 * d].copy_(
                self.visual_init_weight * eye
            )
            self.node_fuse.weight[:, 2 * d :].copy_(self.time_init_weight * eye)
            # Eq. (13) supplies the dominant semantic anchor.  Small non-zero
            # visual/time blocks prevent either enrichment path from being
            # gradient-starved at step zero; per-node norm calibration below
            # makes their scalar initialization comparable across modalities.
            # Eq. (10) only requires W_e to be learnable; the paper does not
            # prescribe W_e = I.  Keep nn.Linear's standard initialization so
            # the initial graph residual is informative but does not begin as
            # a near-duplicate of the local node.
            self.edge_message.reset_parameters()
            # A non-zero ranker lets Eq. (16) update both w and the multimodal
            # node representations from the first backward pass.
            nn.init.xavier_uniform_(self.temporal_ranker.weight)

    @staticmethod
    def _as_int(value):
        if torch.is_tensor(value):
            return int(value.detach().cpu().item())
        return int(value)

    def resolve_task_id(self, task_id):
        task_id = self._as_int(task_id)
        return task_id if task_id in self.task_to_index else None

    def task_index(self, task_id):
        resolved = self.resolve_task_id(task_id)
        if resolved is None:
            return None
        return self.task_to_index.get(resolved)

    def task_buffers(self, task_index):
        return (
            getattr(self, f"source_text_{task_index}"),
            getattr(self, f"source_visual_{task_index}"),
            getattr(self, f"temporal_{task_index}"),
            getattr(self, f"step_ids_{task_index}"),
            getattr(self, f"support_{task_index}"),
        )

    def edge_attributes(self, task_index):
        base = getattr(self, f"edge_attr_{task_index}")
        delta_name = f"edge_temporal_delta_{task_index}"
        if not hasattr(self, delta_name):
            return base
        temporal_mask = base[:, 7:8]
        delta = getattr(self, delta_name)
        epsilon = 1e-4

        # Transition and precedence are probabilities used by Eq. (19), so
        # refine them in logit space and keep them strictly inside [0, 1].
        initial_probabilities = base[:, 2:4].clamp(epsilon, 1.0 - epsilon)
        probability_logits = torch.logit(initial_probabilities)
        probabilities = torch.sigmoid(probability_logits + delta[:, :2])

        # A signed median gap is meaningful (negative means overlapping pseudo
        # spans), whereas MAD is a dispersion and must remain non-negative.
        gap_median = base[:, 4] + delta[:, 2]
        initial_mad = base[:, 5].clamp_min(epsilon)
        mad_unconstrained = torch.log(torch.expm1(initial_mad))
        gap_mad = F.softplus(mad_unconstrained + delta[:, 3])
        temporal = torch.cat(
            [probabilities, gap_median.unsqueeze(1), gap_mad.unsqueeze(1)],
            dim=1,
        )
        temporal = temporal_mask * temporal
        return torch.cat([base[:, :2], temporal, base[:, 6:]], dim=1)

    def encode_task_nodes(
        self,
        task_id,
        text_mapper,
        video_mapper,
        propagate=True,
    ):
        task_index = self.task_index(task_id)
        if task_index is None:
            return None
        source_text, source_visual, temporal, step_ids, support = self.task_buffers(task_index)
        # Eq. (9) constructs a d-dimensional node in the shared encoder
        # space.  Cosine normalization belongs only to Eq. (18) scoring; it
        # must not force the node passed to message passing/association to
        # unit norm.
        semantic_projected = self.semantic_proj(text_mapper(source_text))
        visual_projected = self.visual_proj(video_mapper(source_visual))
        time_projected = self.time_proj(temporal)
        semantic_norm = semantic_projected.detach().norm(dim=1, keepdim=True)
        fusion_visual_projected = (
            F.normalize(visual_projected, p=2, dim=1) * semantic_norm
        )
        fusion_time_projected = (
            F.normalize(time_projected, p=2, dim=1) * semantic_norm
        )
        concatenated = torch.cat(
            [semantic_projected, fusion_visual_projected, fusion_time_projected],
            dim=1,
        )
        local_nodes = self.node_fuse(concatenated)
        semantic = F.normalize(semantic_projected, p=2, dim=1)
        visual = F.normalize(visual_projected, p=2, dim=1)
        time = F.normalize(time_projected, p=2, dim=1)
        refined_nodes = (
            self.propagate(task_index, local_nodes) if propagate else local_nodes
        )
        return {
            "task_id": self.resolve_task_id(task_id),
            "task_index": task_index,
            "semantic": semantic,
            "visual": visual,
            "time": time,
            "semantic_projected": semantic_projected,
            "visual_projected": visual_projected,
            "time_projected": time_projected,
            "visual_fusion_input": fusion_visual_projected,
            "time_fusion_input": fusion_time_projected,
            "local_nodes": local_nodes,
            "nodes": refined_nodes,
            "step_ids": step_ids,
            "support": support,
        }

    def propagate(self, task_index, nodes):
        edge_index = getattr(self, f"edge_index_{task_index}")
        edge_attributes = self.edge_attributes(task_index)
        if edge_index.numel() == 0:
            return nodes
        source, target = edge_index[:, 0], edge_index[:, 1]
        edge_embedding = self.edge_proj(edge_attributes)
        logits = self.edge_score(
            torch.cat([nodes[source], nodes[target], edge_embedding], dim=1)
        ).squeeze(1)
        weights = torch.zeros_like(logits)
        for target_id in torch.unique(target):
            mask = target == target_id
            weights[mask] = torch.softmax(logits[mask], dim=0)
        messages = self.edge_message(nodes[source]) * weights.unsqueeze(1)
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, target, messages)
        if self.graph_update_max_ratio > 0:
            node_norm = nodes.norm(dim=1, keepdim=True).clamp_min(1e-8)
            update_norm = aggregate.norm(dim=1, keepdim=True).clamp_min(1e-8)
            max_update_norm = self.graph_update_max_ratio * node_norm
            aggregate = aggregate * (max_update_norm / update_norm).clamp(max=1.0)
        # Eq. (10): u_tilde_j = u_j + sum_i alpha_{i->j} W_e u_i.
        return nodes + aggregate

    def temporal_relation_matrix(self, task_index, dtype=None):
        edge_index = getattr(self, f"edge_index_{task_index}")
        edge_attributes = self.edge_attributes(task_index)
        node_count = int(getattr(self, f"step_ids_{task_index}").numel())
        matrix = edge_attributes.new_zeros((node_count, node_count))
        if dtype is not None:
            matrix = matrix.to(dtype=dtype)
        if edge_index.numel() > 0:
            source, target = edge_index[:, 0], edge_index[:, 1]
            # Eq. (19) uses the temporal transition probability.
            probabilities = edge_attributes[:, 2].to(matrix.dtype)
            matrix[source, target] = probabilities
        return matrix

    def retrieve_routes(
        self,
        sample,
        text_mapper,
        video_mapper,
    ):
        """Retrieve top-k routes using Eqs. (18-20)."""
        encoded = self.encode_task_nodes(
            sample["cls"],
            text_mapper,
            video_mapper,
            propagate=True,
        )
        if encoded is None:
            return None
        mapped_step_features = sample["step_features"]
        mapped_frame_features = sample["frame_features"]
        # Eq. (18) uses cosine-normalized scoring views.  Eqs. (21-22) must
        # still receive the unnormalized encoder output X^v, not this view.
        step_score_features = F.normalize(
            self.semantic_proj(mapped_step_features), p=2, dim=1
        )
        frame_score_features = F.normalize(
            self.visual_proj(mapped_frame_features), p=2, dim=1
        )

        semantic_scores = step_score_features @ encoded["semantic"].transpose(0, 1)
        frame_node_scores = (
            frame_score_features @ encoded["visual"].transpose(0, 1)
        )
        route_length = int(step_score_features.shape[0])
        frame_count = int(frame_score_features.shape[0])
        step_view = F.normalize(mapped_step_features, p=2, dim=1)
        frame_view = F.normalize(mapped_frame_features, p=2, dim=1)
        alignment_logits = (
            step_view @ frame_view.transpose(0, 1)
        ) / self.retriever_local_alignment_temperature
        frame_positions = (
            torch.arange(
                frame_count,
                device=alignment_logits.device,
                dtype=alignment_logits.dtype,
            )
            + 0.5
        ) / max(frame_count, 1)
        if route_length == 1:
            expected_positions = alignment_logits.new_tensor([0.5])
        else:
            expected_positions = torch.linspace(
                0.0,
                1.0,
                route_length,
                device=alignment_logits.device,
                dtype=alignment_logits.dtype,
            )
        position_delta = (
            frame_positions.unsqueeze(0) - expected_positions.unsqueeze(1)
        ) / self.retriever_local_position_sigma
        alignment_logits = alignment_logits - 0.5 * position_delta.square()
        local_alignment_weights = torch.softmax(alignment_logits, dim=1)
        visual_scores = local_alignment_weights @ frame_node_scores

        node_temporal = getattr(self, f"temporal_{encoded['task_index']}").to(
            dtype=visual_scores.dtype
        )
        node_center = 0.5 * (node_temporal[:, 0] + node_temporal[:, 1])
        node_duration = node_temporal[:, 2].clamp_min(0.0)
        visual_center = local_alignment_weights @ frame_positions
        query_center = 0.5 * visual_center + 0.5 * expected_positions
        query_duration = visual_scores.new_full(
            (route_length,), 1.0 / max(route_length, 1)
        )
        center_similarity = torch.exp(
            -0.5
            * (
                (query_center.unsqueeze(1) - node_center.unsqueeze(0))
                / self.retriever_local_position_sigma
            ).square()
        )
        duration_scale = max(1.0 / max(route_length, 1), 0.1)
        duration_similarity = torch.exp(
            -0.5
            * (
                (query_duration.unsqueeze(1) - node_duration.unsqueeze(0))
                / duration_scale
            ).square()
        )
        temporal_position_scores = 0.75 * center_similarity + 0.25 * duration_similarity

        gate_k = min(self.retriever_local_candidate_topn, semantic_scores.shape[1])
        _, gate_nodes = torch.topk(semantic_scores, k=gate_k, dim=1)
        gate_mask = torch.zeros_like(semantic_scores, dtype=torch.bool)
        gate_mask.scatter_(1, gate_nodes, True)

        def bounded_candidate_residual(scores):
            mask = gate_mask.to(scores.dtype)
            count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (scores * mask).sum(dim=1, keepdim=True) / count
            variance = ((scores - mean).square() * mask).sum(dim=1, keepdim=True) / count
            scale = variance.sqrt().clamp_min(0.05)
            return torch.tanh((scores - mean) / scale)

        node_scores = (
            semantic_scores
            + self.retriever_local_visual_weight
            * bounded_candidate_residual(visual_scores)
            + self.retriever_local_temporal_weight
            * bounded_candidate_residual(temporal_position_scores)
        )
        node_scores = node_scores.masked_fill(~gate_mask, -1.0e4)
        temporal_matrix = self.temporal_relation_matrix(
            encoded["task_index"], dtype=node_scores.dtype
        )

        route_length, node_count = node_scores.shape
        if route_length == 0 or node_count == 0:
            return None
        topk = 5
        beam_size = 20
        gamma = 0.78
        keep = min(max(beam_size, topk), node_count)
        beam_scores, first_nodes = torch.topk(node_scores[0], k=keep)
        beam_routes = first_nodes.unsqueeze(1)
        for position in range(1, route_length):
            previous = beam_routes[:, -1]
            expanded = (
                beam_scores.unsqueeze(1)
                + node_scores[position].unsqueeze(0)
                + float(gamma) * temporal_matrix[previous]
            )
            flat_scores = expanded.reshape(-1)
            keep = min(max(int(beam_size), int(topk)), flat_scores.numel())
            beam_scores, flat_index = torch.topk(flat_scores, k=keep)
            parent = torch.div(flat_index, node_count, rounding_mode="floor")
            next_node = flat_index.remainder(node_count)
            beam_routes = torch.cat(
                [beam_routes[parent], next_node.unsqueeze(1)], dim=1
            )

        route_count = min(int(topk), beam_routes.shape[0])
        routes = beam_routes[:route_count]
        route_scores = beam_scores[:route_count]
        route_nodes = encoded["nodes"][routes]
        hard_route_weights = node_scores.new_zeros((route_length, node_count))
        hard_route_weights.scatter_add_(
            1,
            routes.transpose(0, 1),
            hard_route_weights.new_full(
                (route_length, route_count), 1.0 / route_count
            ),
        )

        soft_weights = []
        previous = None
        for position in range(route_length):
            logits = node_scores[position]
            if previous is not None:
                logits = logits + gamma * (previous @ temporal_matrix)
            current = torch.softmax(logits / self.retrieval_temperature, dim=0)
            soft_weights.append(current)
            previous = current
        soft_weights = torch.stack(soft_weights)
        route_weights = hard_route_weights - soft_weights.detach() + soft_weights
        fused_route = route_weights @ encoded["nodes"]
        return {
            "input_task_id": self._as_int(sample["cls"]),
            "resolved_task_id": encoded["task_id"],
            "routes": routes,
            "route_scores": route_scores,
            "route_nodes": route_nodes,
            "fused_route": fused_route,
            "hard_route_weights": hard_route_weights,
            "soft_route_weights": soft_weights,
            "route_step_ids": encoded["step_ids"][routes],
            "semantic_scores": semantic_scores,
            "visual_scores": visual_scores,
            "temporal_position_scores": temporal_position_scores,
            "local_alignment_weights": local_alignment_weights,
            "node_scores": node_scores,
            "temporal_matrix": temporal_matrix,
            "gamma": float(gamma),
            "mapped_step_features": mapped_step_features,
            "mapped_frame_features": mapped_frame_features,
        }

    def associate_retrieved_route(
        self,
        sample,
        text_mapper,
        video_mapper,
    ):
        retrieval = self.retrieve_routes(sample, text_mapper, video_mapper)
        if retrieval is None:
            return None
        decoded_route, decoded_frames = self.decoder(
            retrieval["fused_route"], retrieval["mapped_frame_features"]
        )
        decoded_similarity = (
            F.normalize(decoded_route, p=2, dim=1)
            @ F.normalize(decoded_frames, p=2, dim=1).transpose(0, 1)
        ) / self.guided_score_temperature
        enhanced_sample = dict(sample)
        enhanced_sample["step_features"] = decoded_route
        enhanced_sample["frame_features"] = decoded_frames
        # Eq. (27) score-only prediction head.  This matrix is shared by
        # guided Drop-DTW, guided clustering and inference.  Normalization is
        # local to score construction and does not overwrite either embedding.
        enhanced_sample["pairwise_scores"] = decoded_similarity
        enhanced_sample["pairwise_score_temperature"] = (
            self.guided_score_temperature
        )
        return {
            "sample": enhanced_sample,
            "retrieval": retrieval,
            "decoded_route": decoded_route,
            "decoded_frames": decoded_frames,
        }

    def _route_node_ids(self, sample, encoded):
        resolved_task_id = int(encoded["task_id"])
        original_task_id = self._as_int(sample["cls"])
        video_id = str(sample["name"])
        sample_step_ids = sample["step_ids"].long()
        node_step_ids = encoded["step_ids"]
        rank_prior = getattr(self, f"rank_prior_{encoded['task_index']}")
        selected = []
        assignment_hits = 0
        route_length = int(sample_step_ids.numel())
        for position, step_id in enumerate(sample_step_ids):
            original_key = (original_task_id, video_id, position)
            resolved_key = (resolved_task_id, video_id, position)
            node_id = self.assignment_to_node.get(original_key)
            if node_id is None:
                node_id = self.assignment_to_node.get(resolved_key)
            if node_id is not None and int(node_step_ids[node_id]) == int(step_id):
                assignment_hits += 1
            else:
                candidates = torch.nonzero(
                    node_step_ids == step_id, as_tuple=False
                ).flatten()
                if candidates.numel() == 0:
                    continue
                normalized_position = position / max(route_length - 1, 1)
                nearest = torch.argmin(
                    (rank_prior[candidates] - normalized_position).abs()
                )
                node_id = int(candidates[nearest])
            selected.append((position, int(node_id)))
        return selected, assignment_hits

    def memory_constraint_loss(
        self,
        samples,
        text_mapper,
        video_mapper,
    ):
        """Original MCL: L_mem = L_sem + lambda * L_temp (Eqs. 14-17).

        Node occurrences are selected by the saved pseudo-alignment assignment,
        so i < j is the actual annotated order of each training video, including
        repeated steps and non-canonical execution routes.
        """
        semantic_losses = []
        refined_semantic_diagnostics = []
        temporal_losses = []
        encoded_tasks = {}
        selected_nodes = 0
        assignment_hits = 0
        temporal_pairs = 0
        temporal_skipped_same_node_pairs = 0
        for sample in samples:
            requested_task = self._as_int(sample["cls"])
            if requested_task not in encoded_tasks:
                encoded_tasks[requested_task] = self.encode_task_nodes(
                    requested_task, text_mapper, video_mapper, propagate=True
                )
            encoded = encoded_tasks[requested_task]
            if encoded is None:
                continue
            route, hits = self._route_node_ids(sample, encoded)
            if not route:
                continue
            positions = torch.tensor(
                [position for position, _ in route],
                dtype=torch.long,
                device=encoded["local_nodes"].device,
            )
            node_ids = torch.tensor(
                [node_id for _, node_id in route],
                dtype=torch.long,
                device=encoded["local_nodes"].device,
            )
            # Eqs. (9-10) distinguish the fused local node u from the
            # graph-refined node u_tilde. Eqs. (14-16) explicitly constrain u.
            local_route_nodes = encoded["local_nodes"][node_ids]
            refined_route_nodes = encoded["nodes"][node_ids]
            sample_text = sample["step_features"][positions]
            # Eq. (14) is squared L2 in the shared embedding space.  Do not
            # replace it with a normalized/cosine surrogate.
            semantic_target = self.semantic_proj(sample_text)
            semantic_losses.append(
                (local_route_nodes - semantic_target).square().sum(dim=1).mean()
            )
            # Monitor graph drift without pulling the graph update back into
            # the text anchor; this value is deliberately excluded from L_mem.
            refined_semantic_diagnostics.append(
                (refined_route_nodes - semantic_target)
                .square()
                .sum(dim=1)
                .mean()
                .detach()
            )

            if local_route_nodes.shape[0] > 1:
                tau = self.temporal_ranker(local_route_nodes).squeeze(1)
                pair_index = torch.triu_indices(
                    tau.numel(), tau.numel(), offset=1, device=tau.device
                )
                different_node = node_ids[pair_index[0]] != node_ids[pair_index[1]]
                temporal_skipped_same_node_pairs += int(
                    (~different_node).sum().detach().cpu()
                )
                if bool(different_node.any()):
                    pair_index = pair_index[:, different_node]
                    if pair_index.numel() == 0:
                        selected_nodes += len(route)
                        assignment_hits += hits
                        continue
                    temporal_losses.append(
                        F.relu(
                            1.0
                            - (tau[pair_index[1]] - tau[pair_index[0]])
                        ).sum()
                    )
                    temporal_pairs += int(pair_index.shape[1])
            selected_nodes += len(route)
            assignment_hits += hits

        reference = next(self.parameters())
        zero = reference.new_zeros(())
        loss_semantic = (
            torch.stack(semantic_losses).mean() if semantic_losses else zero
        )
        loss_temporal = (
            torch.stack(temporal_losses).mean() if temporal_losses else zero
        )
        refined_semantic_diagnostic = (
            torch.stack(refined_semantic_diagnostics).mean()
            if refined_semantic_diagnostics
            else zero.detach()
        )
        loss_total = loss_semantic + 0.5 * loss_temporal
        return {
            "loss": loss_total,
            "semantic": loss_semantic,
            "semantic_refined_diagnostic": refined_semantic_diagnostic,
            "temporal": loss_temporal,
            "temporal_pairs": temporal_pairs,
            "temporal_skipped_same_node_pairs": temporal_skipped_same_node_pairs,
            "samples": len(semantic_losses),
            "selected_nodes": selected_nodes,
            "assignment_hits": assignment_hits,
            "assignment_fallbacks": selected_nodes - assignment_hits,
        }
