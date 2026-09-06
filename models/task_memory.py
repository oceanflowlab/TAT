import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BidirectionalCrossAttentionV2(nn.Module):
    """Light association in Eqs. (21-22), with both branches retained."""

    def __init__(
        self,
        d=512,
        nhead=8,
        dropout=0.1,
        residual_scale_init=1.0,
        learnable_residual_scale=False,
    ):
        super().__init__()
        if not 0.0 < residual_scale_init <= 1.0:
            raise ValueError("residual_scale_init must be in (0, 1]")
        self.route_to_frame = nn.MultiheadAttention(
            d, nhead, dropout=dropout, batch_first=True
        )
        self.frame_to_route = nn.MultiheadAttention(
            d, nhead, dropout=dropout, batch_first=True
        )
        self.route_dropout = nn.Dropout(dropout)
        self.frame_dropout = nn.Dropout(dropout)
        self.route_norm = nn.LayerNorm(d)
        self.frame_norm = nn.LayerNorm(d)
        initial = torch.tensor(float(residual_scale_init))
        if learnable_residual_scale:
            # A sigmoid parameterization guarantees that the applied decoder
            # change never exceeds the standard Transformer candidate.
            epsilon = 1e-6
            initial = initial.clamp(epsilon, 1.0 - epsilon)
            logit = torch.log(initial / (1.0 - initial))
            self.route_residual_scale_logit = nn.Parameter(logit.clone())
            self.frame_residual_scale_logit = nn.Parameter(logit.clone())
        else:
            self.register_buffer("route_residual_scale", initial.clone())
            self.register_buffer("frame_residual_scale", initial.clone())
        self.last_attention_stats = {}

    def residual_scales(self):
        if hasattr(self, "route_residual_scale_logit"):
            return (
                torch.sigmoid(self.route_residual_scale_logit),
                torch.sigmoid(self.frame_residual_scale_logit),
            )
        return self.route_residual_scale, self.frame_residual_scale

    @staticmethod
    def _attention_stats(weights):
        # weights: [batch, heads, queries, keys]
        probabilities = weights.float().clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        key_count = int(probabilities.shape[-1])
        if key_count > 1:
            entropy = entropy / probabilities.new_tensor(key_count).log()
        return {
            "entropy": entropy.mean().detach(),
            "max_probability": probabilities.max(dim=-1).values.mean().detach(),
        }

    def forward(self, route, frames):
        # Eqs. (21-22) specify the two cross-attention directions.  The paper
        # also states that its Transformer follows [40], so retain the
        # standard residual + dropout + LayerNorm wrapper around each
        # attention sublayer.  This preserves input discrimination instead of
        # replacing every feature by a weighted average of the other stream.
        route_update, route_attention = self.route_to_frame(
            query=route,
            key=frames,
            value=frames,
            need_weights=True,
            average_attn_weights=False,
        )
        frame_update, frame_attention = self.frame_to_route(
            query=frames,
            key=route,
            value=route,
            need_weights=True,
            average_attn_weights=False,
        )
        route_stats = self._attention_stats(route_attention)
        frame_stats = self._attention_stats(frame_attention)
        self.last_attention_stats = {
            "route_to_frame_entropy": route_stats["entropy"],
            "route_to_frame_max_probability": route_stats["max_probability"],
            "frame_to_route_entropy": frame_stats["entropy"],
            "frame_to_route_max_probability": frame_stats["max_probability"],
        }
        route_candidate = self.route_norm(
            route + self.route_dropout(route_update)
        )
        frame_candidate = self.frame_norm(
            frames + self.frame_dropout(frame_update)
        )
        route_scale, frame_scale = self.residual_scales()
        enhanced_route = route + route_scale * (route_candidate - route)
        enhanced_frames = frames + frame_scale * (frame_candidate - frames)
        route_denominator = route.detach().norm().clamp_min(1e-8)
        frame_denominator = frames.detach().norm().clamp_min(1e-8)
        self.last_attention_stats.update(
            {
                "route_residual_scale": route_scale.detach(),
                "frame_residual_scale": frame_scale.detach(),
                "route_candidate_delta_ratio": (
                    (route_candidate - route).detach().norm()
                    / route_denominator
                ),
                "frame_candidate_delta_ratio": (
                    (frame_candidate - frames).detach().norm()
                    / frame_denominator
                ),
                "route_applied_delta_ratio": (
                    (enhanced_route - route).detach().norm()
                    / route_denominator
                ),
                "frame_applied_delta_ratio": (
                    (enhanced_frames - frames).detach().norm()
                    / frame_denominator
                ),
            }
        )
        return enhanced_route, enhanced_frames


class ResidualPredictionMapperV2(nn.Module):
    """Modality-specific residual mapper used to form Eq. (27) scores."""

    def __init__(self, d=512):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )
        # Begin very close to the identity map while keeping both residual
        # layers trainable on the first backward pass.
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, features):
        return features + self.residual(features)


class ResidualVisualProjectionV2(nn.Module):
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


class TemporalProjectionV2(nn.Module):
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


class LightCrossModalDecoderV2(nn.Module):
    def __init__(
        self,
        d=512,
        nhead=8,
        num_layers=1,
        dropout=0.1,
        prediction_head="direct",
        residual_scale_init=1.0,
        learnable_residual_scale=False,
    ):
        super().__init__()
        if prediction_head not in {"direct", "dual_residual"}:
            raise ValueError(
                "prediction_head must be 'direct' or 'dual_residual', got "
                f"{prediction_head!r}"
            )
        self.prediction_head = prediction_head
        self.layers = nn.ModuleList(
            [
                BidirectionalCrossAttentionV2(d=d, nhead=nhead, dropout=dropout)
                if not learnable_residual_scale and residual_scale_init == 1.0
                else BidirectionalCrossAttentionV2(
                    d=d,
                    nhead=nhead,
                    dropout=dropout,
                    residual_scale_init=residual_scale_init,
                    learnable_residual_scale=learnable_residual_scale,
                )
                for _ in range(num_layers)
            ]
        )
        # ``direct`` is the paper-faithful control: Eqs. (21-22) feed their
        # Transformer outputs directly to the prediction loss.  Keep the
        # modality-specific residual heads only as an explicit ablation.
        if prediction_head == "dual_residual":
            self.route_prediction_mapper = ResidualPredictionMapperV2(d=d)
            self.frame_prediction_mapper = ResidualPredictionMapperV2(d=d)
        else:
            self.route_prediction_mapper = nn.Identity()
            self.frame_prediction_mapper = nn.Identity()
        self.last_attention_stats = {}

    def forward(self, route, frames):
        route = route.unsqueeze(0)
        frames = frames.unsqueeze(0)
        for layer in self.layers:
            route, frames = layer(route, frames)
        if self.layers:
            self.last_attention_stats = dict(self.layers[-1].last_attention_stats)
        if self.prediction_head == "dual_residual":
            route = self.route_prediction_mapper(route)
            frames = self.frame_prediction_mapper(frames)
            prediction_scale = route.shape[-1] ** 0.5
            route = F.normalize(route, p=2, dim=-1) * prediction_scale
            frames = F.normalize(frames, p=2, dim=-1) * prediction_scale
        return route.squeeze(0), frames.squeeze(0)


class TaskMemory(nn.Module):
    """Multimodal task memory whose semantic anchors use the current encoder.

    Graph files contain encoder-input text/visual prototypes.  The caller
    supplies the current text/video mappings so memory nodes cannot become
    stranded in the feature space of the pseudo-labeling checkpoint.
    """

    def __init__(
        self,
        graph_path,
        assignments_path=None,
        held_out_tasks_csv=None,
        d=512,
        time_dim=3,
        edge_dim=10,
        visual_init_weight=0.05,
        time_init_weight=0.05,
        multimodal_projection_mode="enhanced",
        calibrate_fusion_modalities=True,
        visual_projection_residual_scale=0.1,
        temporal_projection_hidden_dim=128,
        train_temporal_prototypes=True,
        train_temporal_edges=True,
        decoder_nhead=8,
        decoder_layers=1,
        decoder_dropout=0.1,
        decoder_prediction_head="direct",
        decoder_residual_scale_init=0.1,
        learnable_decoder_residual_scale=False,
        guided_score_temperature=0.02,
        guided_score_reference_gamma=None,
        straight_through_retrieval=True,
        retrieval_temperature=1.0,
        association_type="light",
        retriever_log_transition=False,
        transition_log_epsilon=1e-8,
    ):
        super().__init__()
        payload = torch.load(graph_path, map_location="cpu")
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("TaskMemory requires a schema-v2 graph")
        graphs = {int(task_id): graph for task_id, graph in payload["graphs"].items()}
        self.task_ids = sorted(graphs)
        self.task_to_index = {task_id: index for index, task_id in enumerate(self.task_ids)}
        self.global_task_id = None
        self.held_out_to_train = self._load_held_out_mapping(held_out_tasks_csv)
        self.assignment_to_node = self._load_assignments(assignments_path)
        self.visual_init_weight = float(visual_init_weight)
        self.time_init_weight = float(time_init_weight)
        if multimodal_projection_mode != "enhanced":
            raise ValueError(
                "multimodal_projection_mode must be 'enhanced'"
            )
        self.multimodal_projection_mode = multimodal_projection_mode
        self.calibrate_fusion_modalities = bool(calibrate_fusion_modalities)
        self.retriever_log_transition = bool(retriever_log_transition)
        self.transition_log_epsilon = float(transition_log_epsilon)
        self.straight_through_retrieval = bool(straight_through_retrieval)
        self.retrieval_temperature = float(retrieval_temperature)
        self.guided_score_temperature = float(guided_score_temperature)
        self.guided_score_reference_gamma = (
            None
            if guided_score_reference_gamma is None
            else float(guided_score_reference_gamma)
        )
        if association_type != "light":
            raise ValueError(
                "This release keeps only the lightweight TAT association decoder; "
                f"got association_type={association_type!r}"
            )
        self.association_type = association_type
        if self.retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive")
        if self.transition_log_epsilon <= 0:
            raise ValueError("transition_log_epsilon must be positive")
        if self.guided_score_temperature <= 0:
            raise ValueError("guided_score_temperature must be positive")
        if (
            self.guided_score_reference_gamma is not None
            and self.guided_score_reference_gamma <= 0
        ):
            raise ValueError("guided_score_reference_gamma must be positive")
        self.reset_retriever_stats()

        for task_index, task_id in enumerate(self.task_ids):
            graph = graphs[task_id]
            nodes = sorted(graph["nodes"], key=lambda node: int(node["node_id"]))
            raw_text = torch.stack([node["raw_text_proto"].float() for node in nodes])
            raw_visual = torch.stack([node["raw_visual_proto"].float() for node in nodes])
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
            self.register_buffer(f"raw_text_{task_index}", raw_text)
            self.register_buffer(f"raw_visual_{task_index}", raw_visual)
            if train_temporal_prototypes:
                self.register_parameter(
                    f"temporal_{task_index}", nn.Parameter(temporal)
                )
            else:
                self.register_buffer(f"temporal_{task_index}", temporal)
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
            if train_temporal_edges:
                # The four temporal statistics are initialized offline and
                # refined end-to-end. A residual keeps initialization exact;
                # the has-temporal mask prevents absent relations appearing.
                self.register_parameter(
                    f"edge_temporal_delta_{task_index}",
                    nn.Parameter(torch.zeros((edge_attributes.shape[0], 4))),
                )

        self.semantic_proj = nn.Linear(d, d)
        if self.multimodal_projection_mode == "enhanced":
            self.visual_proj = ResidualVisualProjectionV2(
                d=d, residual_scale=visual_projection_residual_scale
            )
            self.time_proj = TemporalProjectionV2(
                time_dim=time_dim,
                hidden_dim=temporal_projection_hidden_dim,
                d=d,
            )
        else:
            self.visual_proj = nn.Linear(d, d)
            self.time_proj = nn.Linear(time_dim, d)
        # Eq. (9): modality-specific projections are concatenated and then
        # projected into the d-dimensional node space.
        self.node_fuse = nn.Linear(3 * d, d)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, d),
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
        self.decoder = LightCrossModalDecoderV2(
            d=d,
            nhead=decoder_nhead,
            num_layers=decoder_layers,
            dropout=decoder_dropout,
            prediction_head=decoder_prediction_head,
            residual_scale_init=decoder_residual_scale_init,
            learnable_residual_scale=learnable_decoder_residual_scale,
        )
        self.reset_parameters()

    @staticmethod
    def _load_held_out_mapping(path):
        if not path:
            return {}
        with open(path, newline="") as handle:
            mapping = {}
            for row in csv.DictReader(handle):
                nearest = row.get("nearest_train_task_id")
                if nearest is not None and nearest != "":
                    mapping[int(row["task_id"])] = int(nearest)
            return mapping

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
        if self.multimodal_projection_mode == "enhanced":
            self.visual_proj.reset_parameters()
            self.time_proj.reset_parameters()
        else:
            nn.init.eye_(self.visual_proj.weight)
            nn.init.zeros_(self.visual_proj.bias)
            nn.init.xavier_uniform_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)
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
        if self.global_task_id is not None:
            return self.global_task_id
        if task_id in self.task_to_index:
            return task_id
        return self.held_out_to_train.get(task_id)

    def task_index(self, task_id):
        resolved = self.resolve_task_id(task_id)
        if resolved is None:
            return None
        return self.task_to_index.get(resolved)

    def reset_retriever_stats(self):
        self.retriever_stats = {
            "calls": 0,
            "routes": 0,
            "positions": 0,
            "top1_temporal_edge_sum": 0.0,
            "top1_temporal_edges": 0,
            "top1_route_score_sum": 0.0,
            "non_same_step_positions": 0,
            "selected_semantic_score_sum": 0.0,
            "selected_visual_score_sum": 0.0,
            "selected_node_score_sum": 0.0,
            "visual_oracle_positions": 0,
            "visual_oracle_top1_hits": 0,
            "visual_oracle_topk_hits": 0,
            "oracle_visual_score_sum": 0.0,
            "visual_oracle_gap_sum": 0.0,
            "unique_top1_nodes_sum": 0.0,
            "graph_update_ratio_sum": 0.0,
            "graph_local_cosine_sum": 0.0,
            "route_input_norm_sum": 0.0,
            "frame_input_norm_sum": 0.0,
            "route_output_norm_sum": 0.0,
            "frame_output_norm_sum": 0.0,
            "route_step_delta_sum": 0.0,
            "frame_delta_sum": 0.0,
            "decoded_similarity_mean_sum": 0.0,
            "decoded_similarity_std_sum": 0.0,
            "decoder_calls": 0,
            "route_to_frame_attention_entropy_sum": 0.0,
            "route_to_frame_attention_max_probability_sum": 0.0,
            "frame_to_route_attention_entropy_sum": 0.0,
            "frame_to_route_attention_max_probability_sum": 0.0,
            "vlm_similarity_mean_sum": 0.0,
            "vlm_similarity_std_sum": 0.0,
            "vlm_precontext_similarity_mean_sum": 0.0,
            "vlm_precontext_similarity_std_sum": 0.0,
            "vlm_route_token_std_sum": 0.0,
            "vlm_frame_token_std_sum": 0.0,
            "route_residual_scale_sum": 0.0,
            "frame_residual_scale_sum": 0.0,
            "route_candidate_delta_ratio_sum": 0.0,
            "frame_candidate_delta_ratio_sum": 0.0,
            "route_applied_delta_ratio_sum": 0.0,
            "frame_applied_delta_ratio_sum": 0.0,
            "semantic_projected_norm_sum": 0.0,
            "visual_projected_norm_sum": 0.0,
            "time_projected_norm_sum": 0.0,
            "semantic_fusion_contribution_norm_sum": 0.0,
            "visual_fusion_contribution_norm_sum": 0.0,
            "time_fusion_contribution_norm_sum": 0.0,
            "local_node_norm_sum": 0.0,
        }

    def pop_retriever_stats(self):
        stats = dict(self.retriever_stats)
        self.reset_retriever_stats()
        return stats

    def task_buffers(self, task_index):
        return (
            getattr(self, f"raw_text_{task_index}"),
            getattr(self, f"raw_visual_{task_index}"),
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
        ablate_node_modality=None,
    ):
        if ablate_node_modality not in {None, "semantic", "visual", "time"}:
            raise ValueError(
                "ablate_node_modality must be semantic, visual, time or None"
            )
        task_index = self.task_index(task_id)
        if task_index is None:
            return None
        raw_text, raw_visual, temporal, step_ids, support = self.task_buffers(task_index)
        # Eq. (9) constructs a d-dimensional node in the shared encoder
        # space.  Cosine normalization belongs only to Eq. (18) scoring; it
        # must not force the node passed to message passing/association to
        # unit norm.
        semantic_raw = self.semantic_proj(text_mapper(raw_text))
        visual_raw = self.visual_proj(video_mapper(raw_visual))
        time_raw = self.time_proj(temporal)
        if self.calibrate_fusion_modalities:
            # Preserve the semantic branch exactly while giving visual and
            # temporal projections a common reference scale before Eq. (9).
            # Detaching the reference norm prevents an artificial gradient
            # from either auxiliary modality into the semantic magnitude.
            semantic_norm = semantic_raw.detach().norm(dim=1, keepdim=True)
            fusion_visual_raw = F.normalize(visual_raw, p=2, dim=1) * semantic_norm
            fusion_time_raw = F.normalize(time_raw, p=2, dim=1) * semantic_norm
        else:
            fusion_visual_raw = visual_raw
            fusion_time_raw = time_raw
        fusion_semantic = (
            torch.zeros_like(semantic_raw)
            if ablate_node_modality == "semantic"
            else semantic_raw
        )
        fusion_visual = (
            torch.zeros_like(fusion_visual_raw)
            if ablate_node_modality == "visual"
            else fusion_visual_raw
        )
        fusion_time = (
            torch.zeros_like(fusion_time_raw)
            if ablate_node_modality == "time"
            else fusion_time_raw
        )
        concatenated = torch.cat(
            [fusion_semantic, fusion_visual, fusion_time], dim=1
        )
        local_nodes = self.node_fuse(concatenated)
        d = semantic_raw.shape[1]
        fuse_weight = self.node_fuse.weight
        semantic_contribution = F.linear(
            semantic_raw, fuse_weight[:, :d], bias=None
        )
        visual_contribution = F.linear(
            fusion_visual_raw, fuse_weight[:, d : 2 * d], bias=None
        )
        time_contribution = F.linear(
            fusion_time_raw, fuse_weight[:, 2 * d :], bias=None
        )
        semantic = F.normalize(semantic_raw, p=2, dim=1)
        visual = F.normalize(visual_raw, p=2, dim=1)
        time = F.normalize(time_raw, p=2, dim=1)
        refined_nodes = (
            self.propagate(task_index, local_nodes) if propagate else local_nodes
        )
        return {
            "task_id": self.resolve_task_id(task_id),
            "task_index": task_index,
            "semantic": semantic,
            "visual": visual,
            "time": time,
            "semantic_raw": semantic_raw,
            "visual_raw": visual_raw,
            "time_raw": time_raw,
            "visual_fusion_input": fusion_visual_raw,
            "time_fusion_input": fusion_time_raw,
            "semantic_contribution": semantic_contribution,
            "visual_contribution": visual_contribution,
            "time_contribution": time_contribution,
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
        # Eq. (10): u_tilde_j = u_j + sum_i alpha_{i->j} W_e u_i.
        return nodes + aggregate

    def temporal_relation_matrix(self, task_index, dtype=None):
        edge_index = getattr(self, f"edge_index_{task_index}")
        edge_attributes = self.edge_attributes(task_index)
        node_count = int(getattr(self, f"step_ids_{task_index}").numel())
        matrix = edge_attributes.new_zeros((node_count, node_count))
        if dtype is not None:
            matrix = matrix.to(dtype=dtype)
        if self.retriever_log_transition:
            matrix.fill_(float(np.log(self.transition_log_epsilon)))
        if edge_index.numel() > 0:
            source, target = edge_index[:, 0], edge_index[:, 1]
            # Eq. (19) uses the temporal transition probability.
            probabilities = edge_attributes[:, 2].to(matrix.dtype)
            if self.retriever_log_transition:
                probabilities = probabilities.clamp_min(
                    self.transition_log_epsilon
                ).log()
            matrix[source, target] = probabilities
        return matrix

    def retrieve_routes(
        self,
        sample,
        text_mapper,
        video_mapper,
        topk=5,
        beam_size=20,
        mu=0.5,
        gamma=1.0,
        sample_features_are_mapped=False,
        propagate_nodes=True,
        ablate_node_modality=None,
    ):
        """Retrieve top-k routes using Eqs. (18-20)."""
        encoded = self.encode_task_nodes(
            sample["cls"],
            text_mapper,
            video_mapper,
            propagate=propagate_nodes,
            ablate_node_modality=ablate_node_modality,
        )
        if encoded is None:
            return None
        mapped_step_features = sample["step_features"]
        mapped_frame_features = sample["frame_features"]
        if not sample_features_are_mapped:
            mapped_step_features = text_mapper(mapped_step_features)
            mapped_frame_features = video_mapper(mapped_frame_features)
        # Eq. (18) uses cosine-normalized scoring views.  Eqs. (21-22) must
        # still receive the unnormalized encoder output X^v, not this view.
        step_score_features = F.normalize(
            self.semantic_proj(mapped_step_features), p=2, dim=1
        )
        frame_score_features = F.normalize(
            self.visual_proj(mapped_frame_features), p=2, dim=1
        )

        semantic_scores = step_score_features @ encoded["semantic"].transpose(0, 1)
        visual_scores = frame_score_features @ encoded["visual"].transpose(0, 1)
        visual_scores = visual_scores.max(dim=0).values
        if ablate_node_modality == "semantic":
            semantic_scores = torch.zeros_like(semantic_scores)
        if ablate_node_modality == "visual":
            visual_scores = torch.zeros_like(visual_scores)
        node_scores = semantic_scores + float(mu) * visual_scores.unsqueeze(0)
        temporal_matrix = self.temporal_relation_matrix(
            encoded["task_index"], dtype=node_scores.dtype
        )

        route_length, node_count = node_scores.shape
        if route_length == 0 or node_count == 0:
            return None
        keep = min(max(int(beam_size), int(topk)), node_count)
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

        if self.straight_through_retrieval:
            # Forward: exactly the hard top-k average in Eq. (20).
            # Backward: a sequential soft relaxation carries L_U gradients
            # into the semantic/visual node scores and temporal edge scores.
            soft_weights = []
            previous = None
            temperature = self.retrieval_temperature
            for position in range(route_length):
                logits = node_scores[position]
                if previous is not None:
                    expected_temporal = previous @ temporal_matrix
                    logits = logits + float(gamma) * expected_temporal
                current = torch.softmax(logits / temperature, dim=0)
                soft_weights.append(current)
                previous = current
            soft_weights = torch.stack(soft_weights)
            route_weights = (
                hard_route_weights - soft_weights.detach() + soft_weights
            )
        else:
            soft_weights = None
            route_weights = hard_route_weights
        fused_route = route_weights @ encoded["nodes"]
        graph_update_ratio = (
            (encoded["nodes"] - encoded["local_nodes"]).norm()
            / encoded["local_nodes"].norm().clamp_min(1e-8)
        )
        graph_local_cosine = F.cosine_similarity(
            encoded["nodes"], encoded["local_nodes"], dim=1
        )
        self.retriever_stats["calls"] += 1
        self.retriever_stats["routes"] += int(route_count)
        self.retriever_stats["positions"] += int(route_length)
        self.retriever_stats["top1_route_score_sum"] += float(
            route_scores[0].detach().cpu()
        )
        top1_route = routes[0]
        positions = torch.arange(route_length, device=routes.device)
        selected_semantic = semantic_scores[positions, top1_route]
        selected_visual = visual_scores[top1_route]
        selected_node = node_scores[positions, top1_route]
        self.retriever_stats["selected_semantic_score_sum"] += float(
            selected_semantic.detach().sum().cpu()
        )
        self.retriever_stats["selected_visual_score_sum"] += float(
            selected_visual.detach().sum().cpu()
        )
        self.retriever_stats["selected_node_score_sum"] += float(
            selected_node.detach().sum().cpu()
        )
        input_step_ids = sample.get("step_ids")
        if input_step_ids is not None and int(input_step_ids.numel()) == route_length:
            input_step_ids = input_step_ids.to(encoded["step_ids"].device).long()
            self.retriever_stats["non_same_step_positions"] += int(
                (encoded["step_ids"][top1_route] != input_step_ids).sum().detach().cpu()
            )
        self.retriever_stats["unique_top1_nodes_sum"] += float(
            torch.unique(top1_route).numel()
        )
        self.retriever_stats["graph_update_ratio_sum"] += float(
            graph_update_ratio.detach().cpu()
        )
        self.retriever_stats["graph_local_cosine_sum"] += float(
            graph_local_cosine.mean().detach().cpu()
        )
        for key, tensor in (
            ("semantic_projected_norm_sum", encoded["semantic_raw"]),
            ("visual_projected_norm_sum", encoded["visual_raw"]),
            ("time_projected_norm_sum", encoded["time_raw"]),
            (
                "semantic_fusion_contribution_norm_sum",
                encoded["semantic_contribution"],
            ),
            (
                "visual_fusion_contribution_norm_sum",
                encoded["visual_contribution"],
            ),
            ("time_fusion_contribution_norm_sum", encoded["time_contribution"]),
            ("local_node_norm_sum", encoded["local_nodes"]),
        ):
            self.retriever_stats[key] += float(
                tensor.norm(dim=1).mean().detach().cpu()
            )

        starts = sample.get("step_starts")
        ends = sample.get("step_ends")
        if starts is not None and ends is not None:
            frame_count = int(frame_score_features.shape[0])
            for position in range(route_length):
                start = max(0, min(self._as_int(starts[position]), frame_count - 1))
                end = max(start, min(self._as_int(ends[position]), frame_count - 1))
                gt_scores = (
                    frame_score_features[start : end + 1]
                    @ encoded["visual"].transpose(0, 1)
                ).max(dim=0).values
                oracle = int(torch.argmax(gt_scores).detach().cpu())
                selected = int(top1_route[position].detach().cpu())
                topk_at_position = routes[:, position]
                self.retriever_stats["visual_oracle_positions"] += 1
                self.retriever_stats["visual_oracle_top1_hits"] += int(selected == oracle)
                self.retriever_stats["visual_oracle_topk_hits"] += int(
                    bool((topk_at_position == oracle).any().detach().cpu())
                )
                selected_gt_score = gt_scores[selected]
                oracle_gt_score = gt_scores[oracle]
                self.retriever_stats["oracle_visual_score_sum"] += float(
                    oracle_gt_score.detach().cpu()
                )
                self.retriever_stats["visual_oracle_gap_sum"] += float(
                    (oracle_gt_score - selected_gt_score).detach().cpu()
                )
        if route_length > 1:
            top_temporal = temporal_matrix[routes[0, :-1], routes[0, 1:]]
            self.retriever_stats["top1_temporal_edge_sum"] += float(
                top_temporal.detach().sum().cpu()
            )
            self.retriever_stats["top1_temporal_edges"] += int(
                top_temporal.numel()
            )
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
            "node_scores": node_scores,
            "temporal_matrix": temporal_matrix,
            "mu": float(mu),
            "gamma": float(gamma),
            "graph_update_ratio": graph_update_ratio,
            "graph_local_cosine_mean": graph_local_cosine.mean(),
            "graph_local_cosine_min": graph_local_cosine.min(),
            "mapped_step_features": mapped_step_features,
            "mapped_frame_features": mapped_frame_features,
        }

    def associate_retrieved_route(
        self,
        sample,
        text_mapper,
        video_mapper,
        topk=5,
        beam_size=20,
        mu=0.5,
        gamma=1.0,
        sample_features_are_mapped=False,
        propagate_nodes=True,
        ablate_node_modality=None,
    ):
        retrieval = self.retrieve_routes(
            sample,
            text_mapper,
            video_mapper,
            topk=topk,
            beam_size=beam_size,
            mu=mu,
            gamma=gamma,
            sample_features_are_mapped=sample_features_are_mapped,
            propagate_nodes=propagate_nodes,
            ablate_node_modality=ablate_node_modality,
        )
        if retrieval is None:
            return None
        decoded_route, decoded_frames = self.decoder(
            retrieval["fused_route"], retrieval["mapped_frame_features"]
        )
        stats = self.retriever_stats
        stats["decoder_calls"] += 1
        attention_stats = self.decoder.last_attention_stats
        attention_stat_keys = {
            "route_to_frame_entropy": "route_to_frame_attention_entropy_sum",
            "route_to_frame_max_probability": "route_to_frame_attention_max_probability_sum",
            "frame_to_route_entropy": "frame_to_route_attention_entropy_sum",
            "frame_to_route_max_probability": "frame_to_route_attention_max_probability_sum",
            "vlm_similarity_mean": "vlm_similarity_mean_sum",
            "vlm_similarity_std": "vlm_similarity_std_sum",
            "vlm_precontext_similarity_mean": "vlm_precontext_similarity_mean_sum",
            "vlm_precontext_similarity_std": "vlm_precontext_similarity_std_sum",
            "vlm_route_token_std": "vlm_route_token_std_sum",
            "vlm_frame_token_std": "vlm_frame_token_std_sum",
            "route_residual_scale": "route_residual_scale_sum",
            "frame_residual_scale": "frame_residual_scale_sum",
            "route_candidate_delta_ratio": "route_candidate_delta_ratio_sum",
            "frame_candidate_delta_ratio": "frame_candidate_delta_ratio_sum",
            "route_applied_delta_ratio": "route_applied_delta_ratio_sum",
            "frame_applied_delta_ratio": "frame_applied_delta_ratio_sum",
        }
        for decoder_key, accumulator_key in attention_stat_keys.items():
            value = attention_stats.get(decoder_key)
            if value is not None:
                stats[accumulator_key] += float(value.detach().cpu())
        stats["route_input_norm_sum"] += float(
            retrieval["fused_route"].norm(dim=1).mean().detach().cpu()
        )
        stats["frame_input_norm_sum"] += float(
            retrieval["mapped_frame_features"].norm(dim=1).mean().detach().cpu()
        )
        stats["route_output_norm_sum"] += float(
            decoded_route.norm(dim=1).mean().detach().cpu()
        )
        stats["frame_output_norm_sum"] += float(
            decoded_frames.norm(dim=1).mean().detach().cpu()
        )
        if retrieval["mapped_step_features"].shape == decoded_route.shape:
            stats["route_step_delta_sum"] += float(
                (decoded_route - retrieval["mapped_step_features"])
                .norm(dim=1)
                .mean()
                .detach()
                .cpu()
            )
        stats["frame_delta_sum"] += float(
            (decoded_frames - retrieval["mapped_frame_features"])
            .norm(dim=1)
            .mean()
            .detach()
            .cpu()
        )
        decoded_similarity = (
            F.normalize(decoded_route, p=2, dim=1)
            @ F.normalize(decoded_frames, p=2, dim=1).transpose(0, 1)
        ) / self.guided_score_temperature
        stats["decoded_similarity_mean_sum"] += float(
            decoded_similarity.mean().detach().cpu()
        )
        stats["decoded_similarity_std_sum"] += float(
            decoded_similarity.std().detach().cpu()
        )
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
        if self.guided_score_reference_gamma is not None:
            enhanced_sample["pairwise_score_reference_gamma"] = (
                self.guided_score_reference_gamma
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
        lambda_temporal=0.5,
        margin=1.0,
        sample_text_is_mapped=False,
    ):
        """Original MCL: L_mem = L_sem + lambda * L_temp (Eqs. 14-17).

        Node occurrences are selected by the saved pseudo-alignment assignment,
        so i < j is the actual annotated order of each training video, including
        repeated steps and non-canonical execution routes.
        """
        semantic_losses = []
        local_semantic_diagnostics = []
        temporal_losses = []
        encoded_tasks = {}
        selected_nodes = 0
        assignment_hits = 0
        temporal_pairs = 0
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
            # Sec. 3.5.5 optimizes the memory after Sec. 3.5.4 graph
            # propagation.  Eq. (14) therefore anchors the graph-refined
            # memory node that is actually consumed by retrieval/alignment.
            # Keeping L_sem on the pre-propagation node would leave the
            # retrieved node free to drift under message passing.
            local_route_nodes = encoded["local_nodes"][node_ids]
            refined_route_nodes = encoded["nodes"][node_ids]
            sample_text = sample["step_features"][positions]
            if not sample_text_is_mapped:
                sample_text = text_mapper(sample_text)
            # Eq. (14) is squared L2 in the shared embedding space.  Do not
            # replace it with a normalized/cosine surrogate.
            semantic_target = self.semantic_proj(sample_text)
            semantic_losses.append(
                (refined_route_nodes - semantic_target).square().sum(dim=1).mean()
            )
            # Diagnostic only: this is deliberately excluded from L_mem.
            local_semantic_diagnostics.append(
                (local_route_nodes - semantic_target)
                .square()
                .sum(dim=1)
                .mean()
                .detach()
            )

            if refined_route_nodes.shape[0] > 1:
                tau = self.temporal_ranker(refined_route_nodes).squeeze(1)
                pair_index = torch.triu_indices(
                    tau.numel(), tau.numel(), offset=1, device=tau.device
                )
                temporal_losses.append(
                    F.relu(
                        float(margin)
                        - (tau[pair_index[1]] - tau[pair_index[0]])
                    ).mean()
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
        local_semantic_diagnostic = (
            torch.stack(local_semantic_diagnostics).mean()
            if local_semantic_diagnostics
            else zero.detach()
        )
        loss_total = loss_semantic + float(lambda_temporal) * loss_temporal
        return {
            "loss": loss_total,
            "semantic": loss_semantic,
            "semantic_local_diagnostic": local_semantic_diagnostic,
            "temporal": loss_temporal,
            "temporal_pairs": temporal_pairs,
            "samples": len(semantic_losses),
            "selected_nodes": selected_nodes,
            "assignment_hits": assignment_hits,
            "assignment_fallbacks": selected_nodes - assignment_hits,
        }
