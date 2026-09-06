import argparse
import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from tqdm import tqdm
from pytorch_lightning.strategies import DDPStrategy

from datasets.batching import unflatten_batch
from datasets.data_module import DataModule
from dp.dp_utils import calibrated_pairwise_scores, compute_all_costs
from dp.exact_dp import crosstask_dp, drop_dtw
from utils.metrics import IoU, framewise_accuracy
from utils.metrics import CLIP_SECONDS
from models.losses import compute_alignment_loss, compute_clust_loss, compute_span_recall_loss
from models.nets import EmbeddingsMapping
from models.task_memory import TaskMemory
from utils.paths import PROJECT_PATH, WEIGHTS_PATH


device = "cuda" if torch.cuda.is_available() else "cpu"
AverageMeter = getattr(torchmetrics, "AverageMeter", torchmetrics.MeanMetric)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="tat_coin")
parser.add_argument(
    "--checkpoint_root",
    type=str,
    default=None,
    help="Physical root directory for checkpoints; defaults to PROJECT_PATH/weights.",
)
parser.add_argument("--dataset", type=str, default="COIN", choices=["COIN"])
parser.add_argument("--seed", type=int, default=40)
parser.add_argument("--memory_graph", type=str, default="outputs/coin_memory/task_memory_graph.pt")
parser.add_argument("--memory_assignments", type=str, default="outputs/coin_memory/task_memory_nodes.pt")
parser.add_argument("--held_out_tasks_csv", type=str, default="local_data/held_out_tasks.csv")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument(
    "--gpus",
    type=int,
    default=1,
    help="Number of visible GPUs. Values above 1 use DDP data parallelism.",
)
parser.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    help="Gradient accumulation; Run3 uses 8 with batch_size 2 to retain effective batch 16.",
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--base_lr", type=float, default=1e-4)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--n_cls", type=int, default=2)

parser.add_argument("--video_layers", type=int, default=2)
parser.add_argument("--text_layers", type=int, default=0)
parser.add_argument("--batchnorm", type=int, default=0)
parser.add_argument("--pretrained_drop", action="store_true", default=False)
parser.add_argument("--init_base_ckpt", type=str, default="weights/drop_dtw_coin/weights-epoch=13.ckpt")
parser.add_argument("--freeze_base_model", action="store_true", default=False)

parser.add_argument("--freeze_memory_prototypes", action="store_true", default=True)
parser.add_argument("--train_memory_prototypes", dest="freeze_memory_prototypes", action="store_false")
parser.add_argument("--freeze_temporal_prototypes", action="store_true", default=False)
parser.add_argument("--train_temporal_prototypes", dest="freeze_temporal_prototypes", action="store_false")
parser.add_argument("--freeze_temporal_edges", action="store_true", default=False)
parser.add_argument("--train_temporal_edges", dest="freeze_temporal_edges", action="store_false")
parser.add_argument("--disable_message_passing", action="store_true", default=False)
parser.add_argument("--decode_frames", action="store_true", default=False)
parser.add_argument("--disable_light_decoder", action="store_true", default=False)
parser.add_argument(
    "--association_type",
    type=str,
    default="light",
    choices=["light"],
    help="Lightweight bidirectional route/frame decoder used by the released TAT setting.",
)
parser.add_argument(
    "--decoder_prediction_head",
    type=str,
    default="direct",
    choices=["direct", "dual_residual"],
    help=(
        "direct uses the bidirectional Transformer outputs from Eqs. (21-22) "
        "directly; dual_residual enables the later mapper ablation."
    ),
)
parser.add_argument(
    "--decoder_residual_scale_init",
    type=float,
    default=0.1,
    help="Initial applied/candidate decoder change ratio in (0, 1].",
)
parser.add_argument(
    "--learnable_decoder_residual_scale",
    action="store_true",
    default=False,
    help="Learn bounded route/frame decoder residual scales via sigmoid.",
)
parser.add_argument("--non_residual_decoder", dest="residual_decoder", action="store_false")
parser.set_defaults(residual_decoder=True)
parser.add_argument("--normalize_enhanced", action="store_true", default=False)
parser.add_argument("--retriever_topk", type=int, default=5)
parser.add_argument("--retriever_beam_size", type=int, default=20)
parser.add_argument("--retriever_mu", type=float, default=0.94)
parser.add_argument("--retriever_gamma", type=float, default=0.78)
parser.add_argument("--retriever_semantic_topk", type=int, default=0)
parser.add_argument("--retriever_same_step_bonus", type=float, default=0.5)
parser.add_argument(
    "--retrieval_gradient",
    type=str,
    default="straight_through",
    choices=["straight_through", "hard"],
)
parser.add_argument("--retrieval_temperature", type=float, default=1.0)
parser.add_argument(
    "--guided_score_temperature",
    type=float,
    default=0.02,
    help="Temperature tau in score-only P_u = cosine(decoded step, frame) / tau.",
)
parser.add_argument(
    "--guided_score_reference_gamma",
    type=float,
    default=0.0,
    help=(
        "Gamma at which guided pairwise_scores are stored. Positive values "
        "rescale P_u by reference_gamma/current_gamma inside Drop-DTW, so "
        "training and evaluation honor the same gamma change as RAW."
    ),
)
parser.add_argument("--memory_init_weight", type=float, default=0.0)
parser.add_argument("--step_delta_init", type=float, default=1.0)
parser.add_argument("--frame_delta_init", type=float, default=1.0)
parser.add_argument("--memory_delta_init_std", type=float, default=0.0)
parser.add_argument("--mem_semantic_mode", type=str, default="normalized_mse", choices=["normalized_mse", "paper_l2"])
parser.add_argument("--mcl_semantic_target", type=str, default="raw", choices=["raw", "refined"])
parser.add_argument("--mcl_temporal_mode", type=str, default="step_mean", choices=["step_mean", "prototype_pair"])
parser.add_argument("--mcl_edge_temporal_loss_mult", type=float, default=0.0)
parser.add_argument("--mcl_edge_margin", type=float, default=1.0)
parser.add_argument("--memory_residual_alpha", type=float, default=1.0)
parser.add_argument("--memory_fusion_mode", type=str, default="gated", choices=["gated", "residual", "concat_mlp"])
parser.add_argument("--memory_as_step", action="store_true", default=False)
parser.add_argument("--memory_as_step_align", action="store_true", default=False)
parser.add_argument("--memory_visual_init_weight", type=float, default=0.05)
parser.add_argument("--memory_time_init_weight", type=float, default=0.05)
parser.add_argument(
    "--memory_projection_mode",
    type=str,
    default="enhanced",
    choices=["enhanced"],
)
parser.add_argument("--memory_visual_projection_residual_scale", type=float, default=0.1)
parser.add_argument("--memory_temporal_hidden_dim", type=int, default=128)
parser.add_argument("--disable_memory_fusion_calibration", action="store_true")
parser.add_argument("--reinit_memory_multimodal", action="store_true", default=False)
parser.add_argument(
    "--retriever_log_transition",
    action="store_true",
    default=False,
    help="Use log transition probabilities in v2 beam route scoring.",
)
parser.add_argument("--transition_log_epsilon", type=float, default=1e-8)

parser.add_argument("--dp_algo", type=str, default="DropDTW", choices=["DropDTW", "OTAM", "NW", "DTW"])
parser.add_argument("--drop_cost", type=str, default="logit", choices=["logit", "learn"])
parser.add_argument("--dtw_softning", type=str, default="prob", choices=["prob", "gamma", "none"])
parser.add_argument("--keep_percentile", type=float, default=0.3)
parser.add_argument("--drop_cost_scale", type=float, default=1.0)
parser.add_argument("--contiguous_drop", type=bool, default=True)
parser.add_argument("--clust_loss_mult", type=float, default=4.0)
parser.add_argument("--dtw_loss_mult", type=float, default=1.0)
parser.add_argument("--dropdtw_align_mult", type=float, default=2.5)
parser.add_argument(
    "--full_dropdtw_loss",
    dest="full_dropdtw_loss",
    action="store_true",
    help="Use the complete Drop-DTW baseline objective for both L_T and L_U (default).",
)
parser.add_argument(
    "--alignment_only_loss",
    dest="full_dropdtw_loss",
    action="store_false",
    help="Ablation: omit the baseline clustering terms from L_T and L_U.",
)
parser.set_defaults(full_dropdtw_loss=True)
parser.add_argument("--mem_loss_mult", type=float, default=1.0)
parser.add_argument(
    "--mem_loss_warmup_epochs",
    type=int,
    default=0,
    help=(
        "Linearly warm the memory/MCL coefficient from 1/warmup_epochs to "
        "--mem_loss_mult; 0 keeps the original constant coefficient."
    ),
)
parser.add_argument("--mem_lambda", type=float, default=0.5)
parser.add_argument("--dtw_xz_gamma", type=float, default=10)
parser.add_argument("--dtw_min_gamma", type=float, default=1)
parser.add_argument("--step_xz_gamma", type=float, default=30)
parser.add_argument("--bg_scope", type=str, default="global", choices=["global", "class", "video"])
parser.add_argument("--align_beta", type=float, default=0.7)
parser.add_argument(
    "--span_recall_loss_mult",
    type=float,
    default=0.0,
    help=(
        "Training-only lightweight span loss.  It uses GT spans in the train "
        "batch to make each step occurrence visible inside its annotated span; "
        "it is never used at test/decode time."
    ),
)
parser.add_argument("--span_recall_topk", type=int, default=3)
parser.add_argument("--span_recall_margin", type=float, default=0.0)
parser.add_argument(
    "--occurrence_temporal_bias",
    type=float,
    default=0.0,
    help=(
        "GT-free decode/eval bias for ordered step occurrences. Positive "
        "values prefer occurrence i near normalized time (i+0.5)/K."
    ),
)
parser.add_argument("--save_every_n_epochs", type=int, default=1)
parser.add_argument("--log_file", type=str, default=None)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--resume_weights_from_checkpoint", type=str, default=None)
parser.add_argument("--gradient_clip_val", type=float, default=1.0)
parser.add_argument("--diagnostic_samples", type=int, default=64)
parser.add_argument(
    "--modality_ablation_samples",
    type=int,
    default=32,
    help="Fixed validation samples for node-modality ablations.",
)
parser.add_argument(
    "--modality_ablation_every_n_epochs",
    type=int,
    default=5,
    help="Run node-modality ablations at epoch 0 and then every N epochs; 0 disables.",
)
parser.add_argument("--limit_train_batches", type=int, default=0)
parser.add_argument("--skip_epoch_eval", action="store_true", default=False)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


class TAMDropDTW(torch.nn.Module):
    def __init__(self, base_model, memory):
        super().__init__()
        self.base_model = base_model
        self.memory = memory

    def map_video(self, x):
        return self.base_model.map_video(x)

    def map_text(self, z):
        return self.base_model.map_text(z)

    def compute_distractors(self, v):
        return self.base_model.compute_distractors(v)

    def enhance_samples(self, samples):
        return [self.enhance_sample(sample) for sample in samples]

    def enhance_sample(self, sample):
        result = self.memory.associate_retrieved_route(
            sample,
            self.map_text,
            self.map_video,
            topk=args.retriever_topk,
            beam_size=args.retriever_beam_size,
            mu=args.retriever_mu,
            gamma=args.retriever_gamma,
            sample_features_are_mapped=True,
            propagate_nodes=not args.disable_message_passing,
        )
        return sample if result is None else result["sample"]

    def memory_constraint_loss(self, samples, lambda_temporal):
        losses = self.memory.memory_constraint_loss(
            samples,
            self.map_text,
            self.map_video,
            lambda_temporal=lambda_temporal,
            sample_text_is_mapped=True,
        )
        losses["edge"] = losses["loss"].new_zeros(())
        return losses


class TrainModule(pl.LightningModule):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.avg_loss_metric = AverageMeter()
        self.last_epoch_record = None
        self.epoch_loss_sums = {}
        self.epoch_loss_count = 0
        self.gradient_snapshot_epoch = -1
        self.gradient_snapshot = {}

    def _effective_mem_loss_mult(self):
        """Epoch-level MCL warm-up; the target objective is unchanged."""
        if args.mem_loss_warmup_epochs <= 0:
            return float(args.mem_loss_mult)
        progress = min(
            1.0,
            float(int(self.current_epoch) + 1) / float(args.mem_loss_warmup_epochs),
        )
        return float(args.mem_loss_mult) * progress

    @staticmethod
    def _module_grad_norm(module):
        squared = None
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            squared = value if squared is None else squared + value
        return 0.0 if squared is None else float(squared.sqrt().cpu())

    @staticmethod
    def _representation_stats(samples):
        """Small, detached batch diagnostic for RAW/TAM score geometry."""
        totals = {
            "similarity_mean": [],
            "similarity_std": [],
            "step_competition": [],
            "step_norm": [],
            "frame_norm": [],
        }
        with torch.no_grad():
            for sample in samples:
                steps = sample["step_features"].detach().float()
                frames = sample["frame_features"].detach().float()
                similarity = sample.get("pairwise_scores")
                if similarity is None:
                    # RAW Drop-DTW consumes dot products divided by gamma_xz;
                    # report that effective score so RAW and score-only P_u
                    # diagnostics are on the same scale.
                    similarity = (
                        steps @ frames.transpose(0, 1)
                    ) / args.dtw_xz_gamma
                else:
                    similarity = similarity.detach().float()
                totals["similarity_mean"].append(similarity.mean())
                totals["similarity_std"].append(
                    similarity.std(unbiased=False)
                )
                totals["step_competition"].append(
                    similarity.std(dim=0, unbiased=False).mean()
                )
                totals["step_norm"].append(steps.norm(dim=1).mean())
                totals["frame_norm"].append(frames.norm(dim=1).mean())
        return {
            name: float(torch.stack(values).mean().cpu())
            for name, values in totals.items()
        }

    def on_after_backward(self):
        """Record one representative gradient snapshot per epoch."""
        epoch = int(self.current_epoch)
        if self.gradient_snapshot_epoch == epoch:
            return
        memory = self.model.memory
        def grad_of(name):
            module = getattr(memory, name, None)
            return 0.0 if module is None else self._module_grad_norm(module)
        snapshot = {
            "grad_base": self._module_grad_norm(self.model.base_model),
            "grad_semantic_projection": grad_of("semantic_proj") or grad_of("text_proj"),
            "grad_visual_projection": grad_of("visual_proj"),
            "grad_time_projection": grad_of("time_proj"),
            "grad_node_fusion": grad_of("node_fuse"),
            "grad_message_passing": grad_of("edge_proj")
            + grad_of("edge_score")
            + grad_of("edge_message"),
            "grad_temporal_ranker": grad_of("temporal_ranker"),
            "grad_decoder": grad_of("decoder"),
        }
        node_fuse = getattr(memory, "node_fuse", None)
        fusion_gradient = (
            None
            if node_fuse is None or getattr(node_fuse, "weight", None) is None
            else node_fuse.weight.grad
        )
        if fusion_gradient is None:
            snapshot.update(
                {
                    "grad_node_fusion_semantic_block": 0.0,
                    "grad_node_fusion_visual_block": 0.0,
                    "grad_node_fusion_time_block": 0.0,
                }
            )
        else:
            semantic_proj = getattr(memory, "semantic_proj", None)
            if semantic_proj is None:
                semantic_proj = getattr(memory, "text_proj", None)
            d = semantic_proj.out_features
            snapshot.update(
                {
                    "grad_node_fusion_semantic_block": float(
                        fusion_gradient[:, :d].detach().norm().cpu()
                    ),
                    "grad_node_fusion_visual_block": float(
                        fusion_gradient[:, d : 2 * d].detach().norm().cpu()
                    ),
                    "grad_node_fusion_time_block": float(
                        fusion_gradient[:, 2 * d :].detach().norm().cpu()
                    ),
                }
            )
        temporal_squared = None
        edge_temporal_squared = None
        for name, parameter in memory.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            if name.startswith("temporal_") and name != "temporal_ranker.weight":
                temporal_squared = value if temporal_squared is None else temporal_squared + value
            elif name.startswith("edge_temporal_delta_"):
                edge_temporal_squared = value if edge_temporal_squared is None else edge_temporal_squared + value
        snapshot["grad_temporal_prototypes"] = (
            0.0 if temporal_squared is None else float(temporal_squared.sqrt().cpu())
        )
        snapshot["grad_temporal_edges"] = (
            0.0 if edge_temporal_squared is None else float(edge_temporal_squared.sqrt().cpu())
        )
        self.gradient_snapshot = snapshot
        self.gradient_snapshot_epoch = epoch

    def configure_optimizers(self):
        if args.base_lr is not None and not args.freeze_base_model:
            base_parameters = [p for p in self.model.base_model.parameters() if p.requires_grad]
            memory_parameters = [
                p for p in self.model.memory.parameters()
                if p.requires_grad
            ]
            param_groups = []
            if base_parameters:
                param_groups.append({"params": base_parameters, "lr": args.base_lr, "name": "base"})
            if memory_parameters:
                param_groups.append({"params": memory_parameters, "lr": args.lr, "name": "memory"})
            optimizer = torch.optim.Adam(param_groups, weight_decay=args.wd)
            print(
                "Using parameter-group learning rates: "
                f"base_lr={args.base_lr}, memory_lr={args.lr}"
            )
        else:
            trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 40, 60, 80], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, flat_batch, batch_id):
        flat_batch["frame_features"] = self.model.map_video(flat_batch["frame_features"])
        flat_batch["step_features"] = self.model.map_text(flat_batch["step_features"])
        raw_samples = unflatten_batch(flat_batch)
        samples = self.model.enhance_samples(raw_samples)

        if args.drop_cost == "learn":
            mean_steps = torch.stack([s["step_features"].mean(0) for s in samples], 0)
            distractors = self.model.compute_distractors(mean_steps)
            raw_mean_steps = torch.stack([s["step_features"].mean(0) for s in raw_samples], 0)
            raw_distractors = self.model.compute_distractors(raw_mean_steps)
        else:
            distractors = None
            raw_distractors = None

        total_loss = flat_batch["frame_features"].new_tensor(0.0)
        loss_record = {}
        effective_mem_loss_mult = self._effective_mem_loss_mult()
        loss_record["effective_mem_loss_mult"] = total_loss.new_tensor(
            effective_mem_loss_mult
        )
        if args.mem_loss_mult > 0:
            memory_losses = self.model.memory_constraint_loss(
                raw_samples, lambda_temporal=args.mem_lambda
            )
            mem_loss = memory_losses["loss"]
            mem_sem_loss = memory_losses["semantic"]
            mem_sem_local_diagnostic = memory_losses.get(
                "semantic_local_diagnostic", mem_sem_loss.detach()
            )
            mem_temp_loss = memory_losses["temporal"]
            mem_edge_temp_loss = memory_losses["edge"]
            self.log("train/mem_loss", mem_loss)
            self.log("train/mem_sem_loss", mem_sem_loss)
            self.log(
                "train/mem_sem_local_diagnostic",
                mem_sem_local_diagnostic,
            )
            self.log("train/mem_temp_loss", mem_temp_loss)
            self.log("train/mem_edge_temp_loss", mem_edge_temp_loss)
            loss_record["mem_loss"] = mem_loss.detach()
            loss_record["mem_sem_loss"] = mem_sem_loss.detach()
            loss_record["mem_sem_local_diagnostic"] = (
                mem_sem_local_diagnostic.detach()
            )
            loss_record["mem_temp_loss"] = mem_temp_loss.detach()
            loss_record["mem_edge_temp_loss"] = mem_edge_temp_loss.detach()
            total_loss += effective_mem_loss_mult * mem_loss

        if args.clust_loss_mult > 0 and not args.full_dropdtw_loss:
            clust_loss = compute_clust_loss(
                samples,
                distractors,
                xz_hard_ratio=1,
                xz_gamma=args.step_xz_gamma,
                frame_gamma=10,
                all_classes_distinct=False,
                bg_scope=args.bg_scope,
            )
            self.log("train/clust_loss", clust_loss)
            loss_record["clust_loss"] = clust_loss.detach()
            total_loss += args.clust_loss_mult * clust_loss

        if args.dtw_loss_mult > 0:
            raw_alignment_loss = compute_alignment_loss(
                raw_samples,
                raw_distractors,
                contiguous=args.contiguous_drop,
                gamma_xz=args.dtw_xz_gamma,
                gamma_min=args.dtw_min_gamma,
                drop_cost_type=args.drop_cost,
                dp_algo=args.dp_algo,
                keep_percentile=args.keep_percentile,
                softning=args.dtw_softning,
                distinct_step_occurrences=True,
                drop_cost_scale=args.drop_cost_scale,
            )
            guided_alignment_loss = compute_alignment_loss(
                samples,
                distractors,
                contiguous=args.contiguous_drop,
                gamma_xz=args.dtw_xz_gamma,
                gamma_min=args.dtw_min_gamma,
                drop_cost_type=args.drop_cost,
                dp_algo=args.dp_algo,
                keep_percentile=args.keep_percentile,
                softning=args.dtw_softning,
                distinct_step_occurrences=True,
                drop_cost_scale=args.drop_cost_scale,
            )
            if args.full_dropdtw_loss:
                raw_clust_loss = compute_clust_loss(
                    raw_samples,
                    raw_distractors,
                    xz_hard_ratio=1,
                    xz_gamma=args.step_xz_gamma,
                    frame_gamma=10,
                    all_classes_distinct=False,
                    bg_scope=args.bg_scope,
                ) if args.clust_loss_mult > 0 else raw_alignment_loss.new_tensor(0.0)
                guided_clust_loss = compute_clust_loss(
                    samples,
                    distractors,
                    xz_hard_ratio=1,
                    xz_gamma=args.step_xz_gamma,
                    frame_gamma=10,
                    all_classes_distinct=False,
                    bg_scope=args.bg_scope,
                ) if args.clust_loss_mult > 0 else guided_alignment_loss.new_tensor(0.0)
                raw_dropdtw_loss = args.dropdtw_align_mult * raw_alignment_loss + args.clust_loss_mult * raw_clust_loss
                guided_dropdtw_loss = args.dropdtw_align_mult * guided_alignment_loss + args.clust_loss_mult * guided_clust_loss
                dtw_loss = args.dtw_loss_mult * (args.align_beta * raw_dropdtw_loss + guided_dropdtw_loss)
                self.log("train/raw_clust_loss", raw_clust_loss)
                self.log("train/guided_clust_loss", guided_clust_loss)
                self.log("train/raw_dropdtw_loss", raw_dropdtw_loss)
                self.log("train/guided_dropdtw_loss", guided_dropdtw_loss)
                loss_record["raw_clust_loss"] = raw_clust_loss.detach()
                loss_record["guided_clust_loss"] = guided_clust_loss.detach()
                loss_record["raw_dropdtw_loss"] = raw_dropdtw_loss.detach()
                loss_record["guided_dropdtw_loss"] = guided_dropdtw_loss.detach()
                loss_record["weighted_LT_alignment"] = (
                    args.dtw_loss_mult
                    * args.align_beta
                    * args.dropdtw_align_mult
                    * raw_alignment_loss
                ).detach()
                loss_record["weighted_LT_clustering"] = (
                    args.dtw_loss_mult
                    * args.align_beta
                    * args.clust_loss_mult
                    * raw_clust_loss
                ).detach()
                loss_record["weighted_LU_alignment"] = (
                    args.dtw_loss_mult
                    * args.dropdtw_align_mult
                    * guided_alignment_loss
                ).detach()
                loss_record["weighted_LU_clustering"] = (
                    args.dtw_loss_mult
                    * args.clust_loss_mult
                    * guided_clust_loss
                ).detach()
            else:
                dtw_loss = args.dtw_loss_mult * (args.align_beta * raw_alignment_loss + guided_alignment_loss)
            self.log("train/raw_dtw_loss", args.dtw_loss_mult * raw_alignment_loss)
            self.log("train/guided_dtw_loss", args.dtw_loss_mult * guided_alignment_loss)
            self.log("train/dtw_loss", dtw_loss)
            loss_record["raw_alignment_loss"] = raw_alignment_loss.detach()
            loss_record["guided_alignment_loss"] = guided_alignment_loss.detach()
            loss_record["dtw_loss"] = dtw_loss.detach()
            total_loss += dtw_loss

        if args.span_recall_loss_mult > 0:
            raw_span_loss = compute_span_recall_loss(
                raw_samples,
                gamma_xz=args.dtw_xz_gamma,
                topk=args.span_recall_topk,
                margin=args.span_recall_margin,
            )
            guided_span_loss = compute_span_recall_loss(
                samples,
                gamma_xz=args.dtw_xz_gamma,
                topk=args.span_recall_topk,
                margin=args.span_recall_margin,
            )
            span_loss = args.align_beta * raw_span_loss + guided_span_loss
            weighted_span_loss = args.span_recall_loss_mult * span_loss
            self.log("train/raw_span_recall_loss", raw_span_loss)
            self.log("train/guided_span_recall_loss", guided_span_loss)
            self.log("train/span_recall_loss", weighted_span_loss)
            loss_record["raw_span_recall_loss"] = raw_span_loss.detach()
            loss_record["guided_span_recall_loss"] = guided_span_loss.detach()
            loss_record["span_recall_loss"] = weighted_span_loss.detach()
            total_loss += weighted_span_loss

        loss_record["total_loss"] = total_loss.detach()
        if not torch.isfinite(total_loss):
            printable = {
                name: float(value.detach().cpu()) if torch.isfinite(value).all() else str(value.detach().cpu())
                for name, value in loss_record.items()
            }
            raise FloatingPointError(f"Non-finite training loss at epoch={self.current_epoch}, batch={batch_id}: {printable}")

        if batch_id % 25 == 0 and self.trainer.is_global_zero:
            live_record = {
                name: round(float(value.detach().cpu()), 6)
                for name, value in loss_record.items()
                if name
                in {
                    "mem_sem_loss",
                    "mem_sem_local_diagnostic",
                    "mem_temp_loss",
                    "raw_clust_loss",
                    "guided_clust_loss",
                    "raw_alignment_loss",
                    "guided_alignment_loss",
                    "raw_dropdtw_loss",
                    "guided_dropdtw_loss",
                    "total_loss",
                }
            }
            print(
                f"LIVE_LOSS epoch={int(self.current_epoch)} batch={batch_id}: "
                + json.dumps(live_record, sort_keys=True)
            )
            raw_representation = self._representation_stats(raw_samples)
            guided_representation = self._representation_stats(samples)
            representation_record = {
                f"raw_{name}": round(value, 6)
                for name, value in raw_representation.items()
            }
            representation_record.update(
                {
                    f"tam_{name}": round(value, 6)
                    for name, value in guided_representation.items()
                }
            )
            print(
                f"LIVE_REP epoch={int(self.current_epoch)} batch={batch_id}: "
                + json.dumps(representation_record, sort_keys=True)
            )
            decoder_stats = getattr(
                getattr(self.model.memory, "decoder", None),
                "last_attention_stats",
                {},
            )
            decoder_record = {
                name: round(float(value.detach().cpu()), 6)
                for name, value in decoder_stats.items()
                if name
                in {
                    "route_residual_scale",
                    "frame_residual_scale",
                    "route_candidate_delta_ratio",
                    "frame_candidate_delta_ratio",
                    "route_applied_delta_ratio",
                    "frame_applied_delta_ratio",
                    "vlm_similarity_mean",
                    "vlm_similarity_std",
                    "vlm_precontext_similarity_mean",
                    "vlm_precontext_similarity_std",
                    "vlm_route_token_std",
                    "vlm_frame_token_std",
                }
            }
            print(
                f"LIVE_DECODER epoch={int(self.current_epoch)} batch={batch_id}: "
                + json.dumps(decoder_record, sort_keys=True)
            )

        for name, value in loss_record.items():
            self.epoch_loss_sums[name] = self.epoch_loss_sums.get(name, 0.0) + float(value.detach().cpu())
        self.epoch_loss_count += 1
        self.log("train/total_loss", self.avg_loss_metric(total_loss))
        return total_loss

    def training_epoch_end(self, training_step_outputs):
        self.model.eval()
        avg_total_loss = self.avg_loss_metric.compute()
        print("Train Total loss: {:.2f}".format(avg_total_loss))
        self.avg_loss_metric.reset()
        if args.skip_epoch_eval:
            self.model.memory.pop_retriever_stats()
            self.last_epoch_record = {
                "epoch": int(self.current_epoch),
                "train_total_loss": float(avg_total_loss.detach().cpu()),
                "smoke_run": True,
            }
            if self.epoch_loss_count:
                for name, value in sorted(self.epoch_loss_sums.items()):
                    self.last_epoch_record[f"train_{name}"] = (
                        value / self.epoch_loss_count
                    )
            self.epoch_loss_sums = {}
            self.epoch_loss_count = 0
            return
        self.model.memory.pop_retriever_stats()
        accuracy_dtw, iou_dtw, recall = evaluate_tam(
            self.data.val_dataset,
            self.model,
            gamma=args.step_xz_gamma,
            drop_cost=args.drop_cost,
            keep_percentile=args.keep_percentile,
            drop_cost_scale=args.drop_cost_scale,
            use_unlabeled=True,
            occurrence_temporal_bias=args.occurrence_temporal_bias,
        )
        retriever_stats = self.model.memory.pop_retriever_stats()
        raw_accuracy_dtw, raw_iou_dtw, raw_recall = evaluate_raw(
            self.data.val_dataset,
            self.model,
            gamma=args.step_xz_gamma,
            drop_cost=args.drop_cost,
            keep_percentile=args.keep_percentile,
            drop_cost_scale=args.drop_cost_scale,
            use_unlabeled=True,
            occurrence_temporal_bias=args.occurrence_temporal_bias,
        )
        retriever_positions = max(int(retriever_stats.get("positions", 0)), 1)
        retriever_non_same_rate = float(retriever_stats.get("non_same_step_positions", 0)) / retriever_positions
        retriever_visual_positions = max(int(retriever_stats.get("visual_oracle_positions", 0)), 1)
        retriever_visual_top1_rate = float(retriever_stats.get("visual_oracle_top1_hits", 0)) / retriever_visual_positions
        retriever_visual_topk_rate = float(retriever_stats.get("visual_oracle_topk_hits", 0)) / retriever_visual_positions
        retriever_selected_visual_avg = float(retriever_stats.get("selected_visual_score_sum", 0.0)) / retriever_visual_positions
        retriever_oracle_visual_avg = float(retriever_stats.get("oracle_visual_score_sum", 0.0)) / retriever_visual_positions
        retriever_visual_gap_avg = float(retriever_stats.get("visual_oracle_gap_sum", 0.0)) / retriever_visual_positions
        retriever_selected_semantic_avg = float(retriever_stats.get("selected_semantic_score_sum", 0.0)) / retriever_visual_positions
        retriever_selected_node_score_avg = float(retriever_stats.get("selected_node_score_sum", 0.0)) / retriever_visual_positions
        retriever_temporal_edges = max(int(retriever_stats.get("top1_temporal_edges", 0)), 1)
        retriever_top1_temporal_edge_avg = float(retriever_stats.get("top1_temporal_edge_sum", 0.0)) / retriever_temporal_edges
        retriever_route_score_avg = float(retriever_stats.get("top1_route_score_sum", 0.0)) / max(int(retriever_stats.get("calls", 0)), 1)
        retriever_calls = max(int(retriever_stats.get("calls", 0)), 1)
        decoder_calls = max(int(retriever_stats.get("decoder_calls", 0)), 1)
        retriever_unique_nodes_avg = float(retriever_stats.get("unique_top1_nodes_sum", 0.0)) / retriever_calls
        graph_update_ratio_avg = float(retriever_stats.get("graph_update_ratio_sum", 0.0)) / retriever_calls
        graph_local_cosine_avg = float(retriever_stats.get("graph_local_cosine_sum", 0.0)) / retriever_calls
        route_input_norm_avg = float(retriever_stats.get("route_input_norm_sum", 0.0)) / decoder_calls
        frame_input_norm_avg = float(retriever_stats.get("frame_input_norm_sum", 0.0)) / decoder_calls
        route_output_norm_avg = float(retriever_stats.get("route_output_norm_sum", 0.0)) / decoder_calls
        frame_output_norm_avg = float(retriever_stats.get("frame_output_norm_sum", 0.0)) / decoder_calls
        route_step_delta_avg = float(retriever_stats.get("route_step_delta_sum", 0.0)) / decoder_calls
        frame_delta_avg = float(retriever_stats.get("frame_delta_sum", 0.0)) / decoder_calls
        decoded_similarity_mean_avg = float(retriever_stats.get("decoded_similarity_mean_sum", 0.0)) / decoder_calls
        decoded_similarity_std_avg = float(retriever_stats.get("decoded_similarity_std_sum", 0.0)) / decoder_calls
        route_attention_entropy_avg = float(retriever_stats.get("route_to_frame_attention_entropy_sum", 0.0)) / decoder_calls
        route_attention_max_avg = float(retriever_stats.get("route_to_frame_attention_max_probability_sum", 0.0)) / decoder_calls
        frame_attention_entropy_avg = float(retriever_stats.get("frame_to_route_attention_entropy_sum", 0.0)) / decoder_calls
        frame_attention_max_avg = float(retriever_stats.get("frame_to_route_attention_max_probability_sum", 0.0)) / decoder_calls
        vlm_similarity_mean_avg = float(retriever_stats.get("vlm_similarity_mean_sum", 0.0)) / decoder_calls
        vlm_similarity_std_avg = float(retriever_stats.get("vlm_similarity_std_sum", 0.0)) / decoder_calls
        vlm_precontext_similarity_mean_avg = float(
            retriever_stats.get("vlm_precontext_similarity_mean_sum", 0.0)
        ) / decoder_calls
        vlm_precontext_similarity_std_avg = float(
            retriever_stats.get("vlm_precontext_similarity_std_sum", 0.0)
        ) / decoder_calls
        vlm_route_token_std_avg = float(
            retriever_stats.get("vlm_route_token_std_sum", 0.0)
        ) / decoder_calls
        vlm_frame_token_std_avg = float(
            retriever_stats.get("vlm_frame_token_std_sum", 0.0)
        ) / decoder_calls
        route_residual_scale_avg = float(
            retriever_stats.get("route_residual_scale_sum", 0.0)
        ) / decoder_calls
        frame_residual_scale_avg = float(
            retriever_stats.get("frame_residual_scale_sum", 0.0)
        ) / decoder_calls
        route_candidate_delta_ratio_avg = float(
            retriever_stats.get("route_candidate_delta_ratio_sum", 0.0)
        ) / decoder_calls
        frame_candidate_delta_ratio_avg = float(
            retriever_stats.get("frame_candidate_delta_ratio_sum", 0.0)
        ) / decoder_calls
        route_applied_delta_ratio_avg = float(
            retriever_stats.get("route_applied_delta_ratio_sum", 0.0)
        ) / decoder_calls
        frame_applied_delta_ratio_avg = float(
            retriever_stats.get("frame_applied_delta_ratio_sum", 0.0)
        ) / decoder_calls
        semantic_projected_norm_avg = float(retriever_stats.get("semantic_projected_norm_sum", 0.0)) / retriever_calls
        visual_projected_norm_avg = float(retriever_stats.get("visual_projected_norm_sum", 0.0)) / retriever_calls
        time_projected_norm_avg = float(retriever_stats.get("time_projected_norm_sum", 0.0)) / retriever_calls
        semantic_fusion_contribution_avg = float(retriever_stats.get("semantic_fusion_contribution_norm_sum", 0.0)) / retriever_calls
        visual_fusion_contribution_avg = float(retriever_stats.get("visual_fusion_contribution_norm_sum", 0.0)) / retriever_calls
        time_fusion_contribution_avg = float(retriever_stats.get("time_fusion_contribution_norm_sum", 0.0)) / retriever_calls
        local_node_norm_avg = float(retriever_stats.get("local_node_norm_sum", 0.0)) / retriever_calls
        print(f"TAM Recall is {recall:.1f}%")
        print(f"TAM DTW Accuracy is {accuracy_dtw:.1f}%")
        print(f"TAM DTW IoU is {iou_dtw:.1f}%")
        print(f"Raw Recall is {raw_recall:.1f}%")
        print(f"Raw DTW Accuracy is {raw_accuracy_dtw:.1f}%")
        print(f"Raw DTW IoU is {raw_iou_dtw:.1f}%")
        print(f"Retriever non-same-step rate is {100.0 * retriever_non_same_rate:.1f}%")
        print(f"Retriever visual oracle top1 is {100.0 * retriever_visual_top1_rate:.1f}%")
        print(f"Retriever visual oracle topk is {100.0 * retriever_visual_topk_rate:.1f}%")
        print(f"Retriever visual gap is {retriever_visual_gap_avg:.4f}")
        print(
            "Retriever scores semantic/visual/node {:.4f}/{:.4f}/{:.4f}, "
            "route unique nodes {:.2f}, temporal edge {:.4f}".format(
                retriever_selected_semantic_avg,
                retriever_selected_visual_avg,
                retriever_selected_node_score_avg,
                retriever_unique_nodes_avg,
                retriever_top1_temporal_edge_avg,
            )
        )
        print(
            "Node projected norms semantic/visual/time {:.2f}/{:.2f}/{:.2f}; "
            "fusion contributions {:.2f}/{:.2f}/{:.2f}; node {:.2f}".format(
                semantic_projected_norm_avg,
                visual_projected_norm_avg,
                time_projected_norm_avg,
                semantic_fusion_contribution_avg,
                visual_fusion_contribution_avg,
                time_fusion_contribution_avg,
                local_node_norm_avg,
            )
        )
        print(
            "Graph update ratio/cosine {:.4f}/{:.4f}; decoder norms "
            "route {:.2f}->{:.2f}, frame {:.2f}->{:.2f}".format(
                graph_update_ratio_avg,
                graph_local_cosine_avg,
                route_input_norm_avg,
                route_output_norm_avg,
                frame_input_norm_avg,
                frame_output_norm_avg,
            )
        )
        print(
            "Decoder deltas route/frame {:.2f}/{:.2f}; decoded sim mean/std "
            "{:.2f}/{:.2f}".format(
                route_step_delta_avg,
                frame_delta_avg,
                decoded_similarity_mean_avg,
                decoded_similarity_std_avg,
            )
        )
        print(
            "Decoder residual scales route/frame {:.4f}/{:.4f}; "
            "candidate delta ratios {:.4f}/{:.4f}; applied delta ratios "
            "{:.4f}/{:.4f}".format(
                route_residual_scale_avg,
                frame_residual_scale_avg,
                route_candidate_delta_ratio_avg,
                frame_candidate_delta_ratio_avg,
                route_applied_delta_ratio_avg,
                frame_applied_delta_ratio_avg,
            )
        )
        print(
            "Attention normalized entropy route->frame/frame->route "
            "{:.4f}/{:.4f}; mean max probability {:.4f}/{:.4f}".format(
                route_attention_entropy_avg,
                frame_attention_entropy_avg,
                route_attention_max_avg,
                frame_attention_max_avg,
            )
        )
        if self.gradient_snapshot:
            print("First-batch gradient norms:", self.gradient_snapshot)
        diagnostics = evaluate_enhancement_diagnostics(
            self.data.val_dataset,
            self.model,
            gamma=args.step_xz_gamma,
            drop_cost=args.drop_cost,
            keep_percentile=args.keep_percentile,
            drop_cost_scale=args.drop_cost_scale,
            max_samples=args.diagnostic_samples,
        )
        if diagnostics:
            print(
                "Diag assignment diff {:.1f}%, TAM labeled {:.1f}% vs Raw {:.1f}%, "
                "GT rank TAM {:.3f} vs Raw {:.3f}, step top1 TAM {:.1f}% vs Raw {:.1f}%".format(
                    100.0 * diagnostics.get("diag_assignment_diff_rate", 0.0),
                    100.0 * diagnostics.get("diag_tam_labeled_frac", 0.0),
                    100.0 * diagnostics.get("diag_raw_labeled_frac", 0.0),
                    diagnostics.get("diag_tam_gt_frame_rank_pct", 0.0),
                    diagnostics.get("diag_raw_gt_frame_rank_pct", 0.0),
                    100.0 * diagnostics.get("diag_tam_gt_step_top1_acc", 0.0),
                    100.0 * diagnostics.get("diag_raw_gt_step_top1_acc", 0.0),
                )
            )
            print(
                "Diag norms step TAM/Raw {:.2f}/{:.2f}, frame TAM/Raw {:.2f}/{:.2f}; "
                "step-competition std TAM/Raw {:.2f}/{:.2f}, temporal std TAM/Raw {:.2f}/{:.2f}".format(
                    diagnostics.get("diag_tam_step_norm", 0.0),
                    diagnostics.get("diag_raw_step_norm", 0.0),
                    diagnostics.get("diag_tam_frame_norm", 0.0),
                    diagnostics.get("diag_raw_frame_norm", 0.0),
                    diagnostics.get("diag_tam_step_competition_std", 0.0),
                    diagnostics.get("diag_raw_step_competition_std", 0.0),
                    diagnostics.get("diag_tam_temporal_sim_std", 0.0),
                    diagnostics.get("diag_raw_temporal_sim_std", 0.0),
                )
            )
        ablation_diagnostics = {}
        ablation_period = args.modality_ablation_every_n_epochs
        if ablation_period > 0 and (
            int(self.current_epoch) == 0
            or int(self.current_epoch) % ablation_period == 0
        ):
            ablation_diagnostics = evaluate_node_modality_ablations(
                self.data.val_dataset,
                self.model,
                gamma=args.step_xz_gamma,
                drop_cost=args.drop_cost,
                keep_percentile=args.keep_percentile,
                drop_cost_scale=args.drop_cost_scale,
                max_samples=args.modality_ablation_samples,
            )
            # Ablation calls are diagnostic only and must not contaminate the
            # next epoch's normal retriever aggregates.
            self.model.memory.pop_retriever_stats()
        if ablation_diagnostics:
            print(
                "Node-modality ablation full IoU {:.2f}; without "
                "semantic/visual/time {:.2f}/{:.2f}/{:.2f}".format(
                    ablation_diagnostics["modality_ablation_full_iou"],
                    ablation_diagnostics[
                        "modality_ablation_without_semantic_iou"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_without_visual_iou"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_without_time_iou"
                    ],
                )
            )
            print(
                "Node-modality IoU drop semantic/visual/time "
                "{:.3f}/{:.3f}/{:.3f}; P_u mean abs change "
                "{:.4f}/{:.4f}/{:.4f}; route change "
                "{:.1%}/{:.1%}/{:.1%}".format(
                    ablation_diagnostics[
                        "modality_ablation_semantic_iou_drop"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_visual_iou_drop"
                    ],
                    ablation_diagnostics["modality_ablation_time_iou_drop"],
                    ablation_diagnostics[
                        "modality_ablation_semantic_pu_mean_abs_change"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_visual_pu_mean_abs_change"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_time_pu_mean_abs_change"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_semantic_route_change_rate"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_visual_route_change_rate"
                    ],
                    ablation_diagnostics[
                        "modality_ablation_time_route_change_rate"
                    ],
                )
            )
        self.log("Metrics/TAMRecall", recall)
        self.log("Metrics/TAMAccuracy", accuracy_dtw)
        self.log("Metrics/TAMIoU", iou_dtw)
        self.log("Metrics/RawRecall", raw_recall)
        self.log("Metrics/RawAccuracy", raw_accuracy_dtw)
        self.log("Metrics/RawIoU", raw_iou_dtw)
        self.log("Metrics/RetrieverNonSameStepRate", retriever_non_same_rate)
        self.log("Metrics/RetrieverVisualOracleTop1", retriever_visual_top1_rate)
        self.log("Metrics/RetrieverVisualOracleTopK", retriever_visual_topk_rate)
        self.log("Metrics/RetrieverVisualGap", retriever_visual_gap_avg)
        for name, value in diagnostics.items():
            self.log(f"Diagnostics/{name}", value)
        for name, value in ablation_diagnostics.items():
            self.log(f"Diagnostics/{name}", value)
        self.last_epoch_record = {
            "epoch": int(self.current_epoch),
            "train_total_loss": float(avg_total_loss.detach().cpu()),
            "tam_recall": float(recall),
            "tam_accuracy": float(accuracy_dtw),
            "tam_iou": float(iou_dtw),
            "raw_recall": float(raw_recall),
            "raw_accuracy": float(raw_accuracy_dtw),
            "raw_iou": float(raw_iou_dtw),
            "retriever_calls": int(retriever_stats.get("calls", 0)),
            "retriever_routes": int(retriever_stats.get("routes", 0)),
            "retriever_positions": int(retriever_stats.get("positions", 0)),
            "retriever_non_same_step_rate": retriever_non_same_rate,
            "retriever_visual_oracle_positions": int(retriever_stats.get("visual_oracle_positions", 0)),
            "retriever_visual_oracle_top1_rate": retriever_visual_top1_rate,
            "retriever_visual_oracle_topk_rate": retriever_visual_topk_rate,
            "retriever_selected_visual_score_avg": retriever_selected_visual_avg,
            "retriever_oracle_visual_score_avg": retriever_oracle_visual_avg,
            "retriever_visual_oracle_gap_avg": retriever_visual_gap_avg,
            "retriever_selected_semantic_score_avg": retriever_selected_semantic_avg,
            "retriever_selected_node_score_avg": retriever_selected_node_score_avg,
            "retriever_top1_temporal_edge_avg": retriever_top1_temporal_edge_avg,
            "retriever_top1_route_score_avg": retriever_route_score_avg,
            "retriever_unique_top1_nodes_avg": retriever_unique_nodes_avg,
            "node_semantic_projected_norm_avg": semantic_projected_norm_avg,
            "node_visual_projected_norm_avg": visual_projected_norm_avg,
            "node_time_projected_norm_avg": time_projected_norm_avg,
            "node_semantic_fusion_contribution_avg": semantic_fusion_contribution_avg,
            "node_visual_fusion_contribution_avg": visual_fusion_contribution_avg,
            "node_time_fusion_contribution_avg": time_fusion_contribution_avg,
            "node_local_norm_avg": local_node_norm_avg,
            "graph_update_ratio_avg": graph_update_ratio_avg,
            "graph_local_cosine_avg": graph_local_cosine_avg,
            "decoder_route_input_norm_avg": route_input_norm_avg,
            "decoder_frame_input_norm_avg": frame_input_norm_avg,
            "decoder_route_output_norm_avg": route_output_norm_avg,
            "decoder_frame_output_norm_avg": frame_output_norm_avg,
            "decoder_route_step_delta_avg": route_step_delta_avg,
            "decoder_frame_delta_avg": frame_delta_avg,
            "decoder_similarity_mean_avg": decoded_similarity_mean_avg,
            "decoder_similarity_std_avg": decoded_similarity_std_avg,
            "route_to_frame_attention_entropy_avg": route_attention_entropy_avg,
            "route_to_frame_attention_max_probability_avg": route_attention_max_avg,
            "frame_to_route_attention_entropy_avg": frame_attention_entropy_avg,
            "frame_to_route_attention_max_probability_avg": frame_attention_max_avg,
            "vlm_similarity_mean_avg": vlm_similarity_mean_avg,
            "vlm_similarity_std_avg": vlm_similarity_std_avg,
            "vlm_precontext_similarity_mean_avg": vlm_precontext_similarity_mean_avg,
            "vlm_precontext_similarity_std_avg": vlm_precontext_similarity_std_avg,
            "vlm_route_token_std_avg": vlm_route_token_std_avg,
            "vlm_frame_token_std_avg": vlm_frame_token_std_avg,
            "decoder_route_residual_scale_avg": route_residual_scale_avg,
            "decoder_frame_residual_scale_avg": frame_residual_scale_avg,
            "decoder_route_candidate_delta_ratio_avg": route_candidate_delta_ratio_avg,
            "decoder_frame_candidate_delta_ratio_avg": frame_candidate_delta_ratio_avg,
            "decoder_route_applied_delta_ratio_avg": route_applied_delta_ratio_avg,
            "decoder_frame_applied_delta_ratio_avg": frame_applied_delta_ratio_avg,
            "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
            "lr_groups": {
                group.get("name", f"group_{idx}"): float(group["lr"])
                for idx, group in enumerate(self.trainer.optimizers[0].param_groups)
            },
        }
        self.last_epoch_record.update(diagnostics)
        self.last_epoch_record.update(ablation_diagnostics)
        self.last_epoch_record.update(self.gradient_snapshot)
        if self.epoch_loss_count:
            for name, value in sorted(self.epoch_loss_sums.items()):
                self.last_epoch_record[f"train_{name}"] = value / self.epoch_loss_count
            total = max(self.last_epoch_record.get("train_total_loss", 0.0), 1e-12)
            effective_mem_mult = self.last_epoch_record.get(
                "train_effective_mem_loss_mult", args.mem_loss_mult
            )
            weighted_mem = effective_mem_mult * self.last_epoch_record.get("train_mem_loss", 0.0)
            if args.full_dropdtw_loss:
                weighted_clust = (
                    self.last_epoch_record.get("train_weighted_LT_clustering", 0.0)
                    + self.last_epoch_record.get("train_weighted_LU_clustering", 0.0)
                )
                weighted_alignment = (
                    self.last_epoch_record.get("train_weighted_LT_alignment", 0.0)
                    + self.last_epoch_record.get("train_weighted_LU_alignment", 0.0)
                )
            else:
                weighted_clust = args.clust_loss_mult * self.last_epoch_record.get("train_clust_loss", 0.0)
                weighted_alignment = self.last_epoch_record.get("train_dtw_loss", 0.0)
            self.last_epoch_record.update(
                {
                    "weighted_mem_contribution": weighted_mem,
                    "weighted_clust_contribution": weighted_clust,
                    "weighted_alignment_contribution": weighted_alignment,
                    "weighted_mem_share": weighted_mem / total,
                    "weighted_clust_share": weighted_clust / total,
                    "weighted_alignment_share": weighted_alignment / total,
                }
            )
            print(
                "Weighted loss contributions mem/clust/alignment: "
                f"{weighted_mem:.4f}/{weighted_clust:.4f}/{weighted_alignment:.4f} "
                f"(shares {weighted_mem / total:.1%}/{weighted_clust / total:.1%}/"
                f"{weighted_alignment / total:.1%})"
            )
            if args.full_dropdtw_loss:
                lt_align = self.last_epoch_record.get("train_weighted_LT_alignment", 0.0)
                lt_clust = self.last_epoch_record.get("train_weighted_LT_clustering", 0.0)
                lu_align = self.last_epoch_record.get("train_weighted_LU_alignment", 0.0)
                lu_clust = self.last_epoch_record.get("train_weighted_LU_clustering", 0.0)
                print(
                    "Objective detail LT-align/LT-clust/LU-align/LU-clust: "
                    f"{lt_align:.4f}/{lt_clust:.4f}/{lu_align:.4f}/{lu_clust:.4f} "
                    f"(shares {lt_align / total:.1%}/{lt_clust / total:.1%}/"
                    f"{lu_align / total:.1%}/{lu_clust / total:.1%})"
                )
        self.epoch_loss_sums = {}
        self.epoch_loss_count = 0


@torch.no_grad()
def step_point_recall(
    sample, step_features, frame_features, pairwise_scores=None
):
    """CrossTask-style recall: one predicted time point per step must fall inside GT span."""
    if pairwise_scores is None:
        pairwise_scores = step_features @ frame_features.T
    text_clip_similarity = pairwise_scores.detach().cpu().numpy()
    optimal_assignment = crosstask_dp(-text_clip_similarity.T).argmax(0)
    detected_steps = 0
    num_steps = int(sample["num_steps"].detach().cpu().numpy())
    for step_idx in range(num_steps):
        start = float(sample["step_starts_sec"][step_idx].detach().cpu().numpy())
        end = float(sample["step_ends_sec"][step_idx].detach().cpu().numpy())
        inferred_time = (int(optimal_assignment[step_idx]) + 0.5) * CLIP_SECONDS
        detected_steps += int(start <= inferred_time <= end)
    return detected_steps, num_steps


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def evaluate_enhancement_diagnostics(
    dataset,
    model,
    gamma,
    drop_cost,
    keep_percentile,
    drop_cost_scale=1.0,
    max_samples=64,
):
    if max_samples <= 0:
        return {}

    model_device = next(model.parameters()).device
    stats = {
        "assignment_diff": [],
        "raw_labeled_frac": [],
        "tam_labeled_frac": [],
        "gt_labeled_frac": [],
        "raw_pred_len_ratio": [],
        "tam_pred_len_ratio": [],
        "raw_gt_pos_sim": [],
        "tam_gt_pos_sim": [],
        "raw_gt_neg_sim": [],
        "tam_gt_neg_sim": [],
        "raw_sim_mean": [],
        "tam_sim_mean": [],
        "raw_sim_std": [],
        "tam_sim_std": [],
        "raw_sim_max": [],
        "tam_sim_max": [],
        "raw_step_competition_std": [],
        "tam_step_competition_std": [],
        "raw_temporal_sim_std": [],
        "tam_temporal_sim_std": [],
        "raw_step_norm": [],
        "tam_step_norm": [],
        "raw_frame_norm": [],
        "tam_frame_norm": [],
        "raw_match_cost": [],
        "tam_match_cost": [],
        "raw_drop_cost": [],
        "tam_drop_cost": [],
        "raw_gt_frame_rank_pct": [],
        "tam_gt_frame_rank_pct": [],
        "raw_gt_step_top1_acc": [],
        "tam_gt_step_top1_acc": [],
        "raw_gt_step_prob": [],
        "tam_gt_step_prob": [],
        "step_delta_norm": [],
        "frame_delta_norm": [],
    }

    iterator = tqdm(dataset, desc="Diagnostics", dynamic_ncols=True, total=min(max_samples, len(dataset)))
    used = 0
    for sample in iterator:
        if used >= max_samples:
            break
        if sample["num_steps"] < 1:
            continue

        raw_sample = dict(sample)
        raw_sample["frame_features"] = model.map_video(sample["frame_features"].to(model_device)).detach().cpu()
        raw_sample["step_features"] = model.map_text(sample["step_features"].to(model_device)).detach().cpu()
        tam_sample = model.enhance_sample(
            {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in raw_sample.items()}
        )
        tam_sample = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in tam_sample.items()}

        if drop_cost == "learn":
            raw_distractor = model.compute_distractors(raw_sample["step_features"].mean(0).to(model_device)).detach().cpu()
            tam_distractor = model.compute_distractors(tam_sample["step_features"].mean(0).to(model_device)).detach().cpu()
        else:
            raw_distractor = None
            tam_distractor = None

        raw_zx, raw_drop, _ = compute_all_costs(
            raw_sample,
            raw_distractor,
            gamma,
            drop_cost_type=drop_cost,
            keep_percentile=keep_percentile,
            distinct_step_occurrences=True,
        )
        raw_drop = raw_drop * float(drop_cost_scale)
        tam_zx, tam_drop, _ = compute_all_costs(
            tam_sample,
            tam_distractor,
            gamma,
            drop_cost_type=drop_cost,
            keep_percentile=keep_percentile,
            distinct_step_occurrences=True,
        )
        tam_drop = tam_drop * float(drop_cost_scale)
        raw_assignment = drop_dtw(raw_zx.detach().cpu().numpy(), raw_drop.detach().cpu().numpy(), return_labels=True) - 1
        tam_assignment = drop_dtw(tam_zx.detach().cpu().numpy(), tam_drop.detach().cpu().numpy(), return_labels=True) - 1
        valid_len = min(len(raw_assignment), len(tam_assignment))
        if valid_len:
            stats["assignment_diff"].append(float(np.mean(raw_assignment[:valid_len] != tam_assignment[:valid_len])))
            stats["raw_labeled_frac"].append(float(np.mean(raw_assignment[:valid_len] >= 0)))
            stats["tam_labeled_frac"].append(float(np.mean(tam_assignment[:valid_len] >= 0)))

        num_frames = int(sample["num_frames"].detach().cpu().numpy())
        num_steps = int(sample["num_steps"].detach().cpu().numpy())
        gt_mask = np.zeros(num_frames, dtype=bool)
        for step_idx in range(num_steps):
            start = int(sample["step_starts"][step_idx].detach().cpu().numpy())
            end = int(sample["step_ends"][step_idx].detach().cpu().numpy())
            gt_mask[start : end + 1] = True
        if num_frames:
            stats["gt_labeled_frac"].append(float(gt_mask.mean()))
        if num_steps and valid_len:
            stats["raw_pred_len_ratio"].append(float(np.mean([np.mean(raw_assignment[:valid_len] == i) for i in range(num_steps)])))
            stats["tam_pred_len_ratio"].append(float(np.mean([np.mean(tam_assignment[:valid_len] == i) for i in range(num_steps)])))

        # Report the score that evaluation actually passes to its softmax.
        # Multiplying by a positive constant leaves rank metrics unchanged,
        # while probability/competition diagnostics become comparable to P_u.
        raw_sim = (
            raw_sample["step_features"] @ raw_sample["frame_features"].T
        ).div(float(gamma)).detach().cpu()
        tam_sim = calibrated_pairwise_scores(tam_sample, gamma)
        if tam_sim is None:
            tam_sim = tam_sample["step_features"] @ tam_sample["frame_features"].T
        tam_sim = tam_sim.detach().cpu()
        stats["raw_sim_mean"].append(float(raw_sim.mean()))
        stats["tam_sim_mean"].append(float(tam_sim.mean()))
        stats["raw_sim_std"].append(float(raw_sim.std()))
        stats["tam_sim_std"].append(float(tam_sim.std()))
        stats["raw_sim_max"].append(float(raw_sim.max()))
        stats["tam_sim_max"].append(float(tam_sim.max()))
        stats["raw_step_competition_std"].append(float(raw_sim.std(dim=0).mean()))
        stats["tam_step_competition_std"].append(float(tam_sim.std(dim=0).mean()))
        stats["raw_temporal_sim_std"].append(float(raw_sim.std(dim=1).mean()))
        stats["tam_temporal_sim_std"].append(float(tam_sim.std(dim=1).mean()))
        stats["raw_step_norm"].append(float(raw_sample["step_features"].norm(dim=1).mean()))
        stats["tam_step_norm"].append(float(tam_sample["step_features"].norm(dim=1).mean()))
        stats["raw_frame_norm"].append(float(raw_sample["frame_features"].norm(dim=1).mean()))
        stats["tam_frame_norm"].append(float(tam_sample["frame_features"].norm(dim=1).mean()))
        pos_raw, pos_tam, neg_raw, neg_tam = [], [], [], []
        raw_rank_scores, tam_rank_scores = [], []
        raw_top1_correct, tam_top1_correct = [], []
        raw_gt_probs, tam_gt_probs = [], []
        raw_step_pred = raw_sim.argmax(dim=0)
        tam_step_pred = tam_sim.argmax(dim=0)
        raw_step_prob = torch.softmax(raw_sim, dim=0)
        tam_step_prob = torch.softmax(tam_sim, dim=0)
        for step_idx in range(num_steps):
            start = int(sample["step_starts"][step_idx].detach().cpu().numpy())
            end = int(sample["step_ends"][step_idx].detach().cpu().numpy())
            if start <= end and end < raw_sim.shape[1] and step_idx < raw_sim.shape[0]:
                span = slice(start, end + 1)
                pos_raw.append(float(raw_sim[step_idx, span].mean()))
                pos_tam.append(float(tam_sim[step_idx, span].mean()))
                neg_mask = torch.ones(raw_sim.shape[1], dtype=torch.bool)
                neg_mask[start : end + 1] = False
                if torch.any(neg_mask):
                    neg_raw.append(float(raw_sim[step_idx, neg_mask].mean()))
                    neg_tam.append(float(tam_sim[step_idx, neg_mask].mean()))

                raw_values = raw_sim[step_idx]
                tam_values = tam_sim[step_idx]
                raw_gt_values = raw_values[span]
                tam_gt_values = tam_values[span]
                if raw_values.numel() > 1 and raw_gt_values.numel() > 0:
                    raw_ranks = (raw_values.unsqueeze(0) > raw_gt_values.unsqueeze(1)).float().mean(dim=1)
                    tam_ranks = (tam_values.unsqueeze(0) > tam_gt_values.unsqueeze(1)).float().mean(dim=1)
                    raw_rank_scores.append(float(raw_ranks.mean()))
                    tam_rank_scores.append(float(tam_ranks.mean()))
                raw_top1_correct.append(float((raw_step_pred[span] == step_idx).float().mean()))
                tam_top1_correct.append(float((tam_step_pred[span] == step_idx).float().mean()))
                raw_gt_probs.append(float(raw_step_prob[step_idx, span].mean()))
                tam_gt_probs.append(float(tam_step_prob[step_idx, span].mean()))
        if pos_raw:
            stats["raw_gt_pos_sim"].append(float(np.mean(pos_raw)))
            stats["tam_gt_pos_sim"].append(float(np.mean(pos_tam)))
        if neg_raw:
            stats["raw_gt_neg_sim"].append(float(np.mean(neg_raw)))
            stats["tam_gt_neg_sim"].append(float(np.mean(neg_tam)))
        if raw_rank_scores:
            stats["raw_gt_frame_rank_pct"].append(float(np.mean(raw_rank_scores)))
            stats["tam_gt_frame_rank_pct"].append(float(np.mean(tam_rank_scores)))
        if raw_top1_correct:
            stats["raw_gt_step_top1_acc"].append(float(np.mean(raw_top1_correct)))
            stats["tam_gt_step_top1_acc"].append(float(np.mean(tam_top1_correct)))
        if raw_gt_probs:
            stats["raw_gt_step_prob"].append(float(np.mean(raw_gt_probs)))
            stats["tam_gt_step_prob"].append(float(np.mean(tam_gt_probs)))

        stats["raw_match_cost"].append(float(raw_zx.mean()))
        stats["tam_match_cost"].append(float(tam_zx.mean()))
        stats["raw_drop_cost"].append(float(raw_drop.mean()))
        stats["tam_drop_cost"].append(float(tam_drop.mean()))
        stats["step_delta_norm"].append(float((tam_sample["step_features"] - raw_sample["step_features"]).norm(dim=1).mean()))
        stats["frame_delta_norm"].append(float((tam_sample["frame_features"] - raw_sample["frame_features"]).norm(dim=1).mean()))
        used += 1

    return {
        "diag_samples": int(used),
        "diag_assignment_diff_rate": _safe_mean(stats["assignment_diff"]),
        "diag_raw_labeled_frac": _safe_mean(stats["raw_labeled_frac"]),
        "diag_tam_labeled_frac": _safe_mean(stats["tam_labeled_frac"]),
        "diag_gt_labeled_frac": _safe_mean(stats["gt_labeled_frac"]),
        "diag_raw_pred_len_ratio": _safe_mean(stats["raw_pred_len_ratio"]),
        "diag_tam_pred_len_ratio": _safe_mean(stats["tam_pred_len_ratio"]),
        "diag_raw_gt_pos_sim": _safe_mean(stats["raw_gt_pos_sim"]),
        "diag_tam_gt_pos_sim": _safe_mean(stats["tam_gt_pos_sim"]),
        "diag_raw_gt_neg_sim": _safe_mean(stats["raw_gt_neg_sim"]),
        "diag_tam_gt_neg_sim": _safe_mean(stats["tam_gt_neg_sim"]),
        "diag_raw_pos_neg_margin": _safe_mean(stats["raw_gt_pos_sim"]) - _safe_mean(stats["raw_gt_neg_sim"]),
        "diag_tam_pos_neg_margin": _safe_mean(stats["tam_gt_pos_sim"]) - _safe_mean(stats["tam_gt_neg_sim"]),
        "diag_raw_sim_mean": _safe_mean(stats["raw_sim_mean"]),
        "diag_tam_sim_mean": _safe_mean(stats["tam_sim_mean"]),
        "diag_raw_sim_std": _safe_mean(stats["raw_sim_std"]),
        "diag_tam_sim_std": _safe_mean(stats["tam_sim_std"]),
        "diag_raw_sim_max": _safe_mean(stats["raw_sim_max"]),
        "diag_tam_sim_max": _safe_mean(stats["tam_sim_max"]),
        "diag_raw_step_competition_std": _safe_mean(stats["raw_step_competition_std"]),
        "diag_tam_step_competition_std": _safe_mean(stats["tam_step_competition_std"]),
        "diag_raw_temporal_sim_std": _safe_mean(stats["raw_temporal_sim_std"]),
        "diag_tam_temporal_sim_std": _safe_mean(stats["tam_temporal_sim_std"]),
        "diag_raw_step_norm": _safe_mean(stats["raw_step_norm"]),
        "diag_tam_step_norm": _safe_mean(stats["tam_step_norm"]),
        "diag_raw_frame_norm": _safe_mean(stats["raw_frame_norm"]),
        "diag_tam_frame_norm": _safe_mean(stats["tam_frame_norm"]),
        "diag_raw_match_cost": _safe_mean(stats["raw_match_cost"]),
        "diag_tam_match_cost": _safe_mean(stats["tam_match_cost"]),
        "diag_raw_drop_cost": _safe_mean(stats["raw_drop_cost"]),
        "diag_tam_drop_cost": _safe_mean(stats["tam_drop_cost"]),
        "diag_raw_gt_frame_rank_pct": _safe_mean(stats["raw_gt_frame_rank_pct"]),
        "diag_tam_gt_frame_rank_pct": _safe_mean(stats["tam_gt_frame_rank_pct"]),
        "diag_raw_gt_step_top1_acc": _safe_mean(stats["raw_gt_step_top1_acc"]),
        "diag_tam_gt_step_top1_acc": _safe_mean(stats["tam_gt_step_top1_acc"]),
        "diag_raw_gt_step_prob": _safe_mean(stats["raw_gt_step_prob"]),
        "diag_tam_gt_step_prob": _safe_mean(stats["tam_gt_step_prob"]),
        "diag_step_delta_norm": _safe_mean(stats["step_delta_norm"]),
        "diag_frame_delta_norm": _safe_mean(stats["frame_delta_norm"]),
    }


@torch.no_grad()
def evaluate_node_modality_ablations(
    dataset,
    model,
    gamma,
    drop_cost,
    keep_percentile,
    drop_cost_scale=1.0,
    max_samples=32,
):
    """Inference-time ablations of Eq. (9) node modalities.

    Temporal graph edges are intentionally retained when the node time prior
    is removed.  This isolates the three node inputs from edge knowledge and
    does not mutate parameters or the normal training forward path.
    """
    if max_samples <= 0 or False:
        return {}

    model_device = next(model.parameters()).device
    variants = ("full", "without_semantic", "without_visual", "without_time")
    metrics = {
        variant: {
            "accuracy": [],
            "iou": [],
            "detected": 0,
            "steps": 0,
            "competition": [],
        }
        for variant in variants
    }
    changes = {
        modality: {"route": [], "pu": []}
        for modality in ("semantic", "visual", "time")
    }

    used = 0
    iterator = tqdm(
        dataset,
        desc="Node-modality ablations",
        dynamic_ncols=True,
        total=min(max_samples, len(dataset)),
    )
    for sample in iterator:
        if used >= max_samples:
            break
        if sample["num_steps"] < 1:
            continue

        mapped = dict(sample)
        mapped["frame_features"] = model.map_video(
            sample["frame_features"].to(model_device)
        )
        mapped["step_features"] = model.map_text(
            sample["step_features"].to(model_device)
        )
        mapped = {
            key: (value.to(model_device) if torch.is_tensor(value) else value)
            for key, value in mapped.items()
        }

        associations = {}
        for variant in variants:
            modality = None if variant == "full" else variant[len("without_") :]
            associations[variant] = model.memory.associate_retrieved_route(
                mapped,
                model.map_text,
                model.map_video,
                topk=args.retriever_topk,
                beam_size=args.retriever_beam_size,
                mu=args.retriever_mu,
                gamma=args.retriever_gamma,
                sample_features_are_mapped=True,
                ablate_node_modality=modality,
            )
        if any(result is None for result in associations.values()):
            continue

        full_result = associations["full"]
        full_pu = full_result["sample"]["pairwise_scores"].detach().cpu()
        full_route = full_result["retrieval"]["routes"][0].detach().cpu()

        for variant, result in associations.items():
            work_sample = {
                key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for key, value in result["sample"].items()
            }
            if drop_cost == "learn":
                distractor = model.compute_distractors(
                    result["sample"]["step_features"].mean(0)
                ).detach().cpu()
            else:
                distractor = None
            zx_costs, drop_costs, _ = compute_all_costs(
                work_sample,
                distractor,
                gamma,
                drop_cost_type=drop_cost,
                keep_percentile=keep_percentile,
                distinct_step_occurrences=True,
            )
            drop_costs = drop_costs * float(drop_cost_scale)
            assignment = drop_dtw(
                zx_costs.detach().cpu().numpy(),
                drop_costs.detach().cpu().numpy(),
                return_labels=True,
            ) - 1
            metrics[variant]["accuracy"].append(
                framewise_accuracy(assignment, sample, use_unlabeled=True)
            )
            metrics[variant]["iou"].append(IoU(assignment, sample))
            detected, steps = step_point_recall(
                sample,
                work_sample["step_features"],
                work_sample["frame_features"],
                work_sample["pairwise_scores"],
            )
            metrics[variant]["detected"] += detected
            metrics[variant]["steps"] += steps
            metrics[variant]["competition"].append(
                float(
                    work_sample["pairwise_scores"]
                    .std(dim=0, unbiased=False)
                    .mean()
                )
            )

            if variant != "full":
                modality = variant[len("without_") :]
                ablated_pu = work_sample["pairwise_scores"]
                ablated_route = result["retrieval"]["routes"][0].detach().cpu()
                changes[modality]["pu"].append(
                    float((ablated_pu - full_pu).abs().mean())
                )
                changes[modality]["route"].append(
                    float((ablated_route != full_route).float().mean())
                )
        used += 1

    result = {"modality_ablation_samples": int(used)}
    for variant in variants:
        steps = metrics[variant]["steps"]
        result[f"modality_ablation_{variant}_accuracy"] = (
            100.0 * _safe_mean(metrics[variant]["accuracy"])
        )
        result[f"modality_ablation_{variant}_iou"] = (
            100.0 * _safe_mean(metrics[variant]["iou"])
        )
        result[f"modality_ablation_{variant}_recall"] = (
            100.0 * metrics[variant]["detected"] / steps if steps else 0.0
        )
        result[f"modality_ablation_{variant}_competition"] = _safe_mean(
            metrics[variant]["competition"]
        )
    full_iou = result["modality_ablation_full_iou"]
    for modality in ("semantic", "visual", "time"):
        without = f"without_{modality}"
        result[f"modality_ablation_{modality}_iou_drop"] = (
            full_iou - result[f"modality_ablation_{without}_iou"]
        )
        result[f"modality_ablation_{modality}_route_change_rate"] = _safe_mean(
            changes[modality]["route"]
        )
        result[f"modality_ablation_{modality}_pu_mean_abs_change"] = _safe_mean(
            changes[modality]["pu"]
        )
    return result


@torch.no_grad()
def _add_occurrence_temporal_bias(zx_costs, strength):
    if strength <= 0:
        return zx_costs
    biased = zx_costs.copy()
    num_steps, num_frames = biased.shape
    if num_steps <= 0 or num_frames <= 0:
        return biased
    frame_pos = (np.arange(num_frames, dtype=np.float32) + 0.5) / float(num_frames)
    for step_idx in range(num_steps):
        expected = (float(step_idx) + 0.5) / float(num_steps)
        biased[step_idx] += float(strength) * np.abs(frame_pos - expected)
    return biased


@torch.no_grad()
def evaluate_tam(
    dataset,
    model,
    gamma,
    drop_cost,
    keep_percentile,
    drop_cost_scale=1.0,
    use_unlabeled=True,
    occurrence_temporal_bias=0.0,
):
    accuracy = 0.0
    iou = 0.0
    detected_steps = 0
    total_steps = 0
    num_samples = 0
    model_device = next(model.parameters()).device
    iterator = tqdm(dataset, desc="Evaluating TAM", dynamic_ncols=True)
    for sample in iterator:
        if sample["num_steps"] < 1:
            continue
        frame_features = model.map_video(sample["frame_features"].to(model_device)).detach().cpu()
        step_features = model.map_text(sample["step_features"].to(model_device)).detach().cpu()
        work_sample = dict(sample)
        work_sample["frame_features"] = frame_features
        work_sample["step_features"] = step_features
        work_sample = model.enhance_sample({k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in work_sample.items()})
        work_sample = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in work_sample.items()}
        if drop_cost == "learn":
            distractor = model.compute_distractors(work_sample["step_features"].mean(0).to(model_device)).detach().cpu()
        else:
            distractor = None
        zx_costs, drop_costs, _ = compute_all_costs(
            work_sample,
            distractor,
            gamma,
            drop_cost_type=drop_cost,
            keep_percentile=keep_percentile,
            distinct_step_occurrences=True,
        )
        drop_costs = drop_costs * float(drop_cost_scale)
        zx_np = zx_costs.detach().cpu().numpy()
        zx_decode = _add_occurrence_temporal_bias(zx_np, occurrence_temporal_bias)
        assignment = drop_dtw(
            zx_decode,
            drop_costs.detach().cpu().numpy(),
            return_labels=True,
        ) - 1
        accuracy += framewise_accuracy(assignment, sample, use_unlabeled=use_unlabeled)
        iou += IoU(assignment, sample)
        sample_detected, sample_steps = step_point_recall(
            sample,
            work_sample["step_features"],
            work_sample["frame_features"],
            work_sample.get("pairwise_scores"),
        )
        detected_steps += sample_detected
        total_steps += sample_steps
        num_samples += 1
    recall = 100.0 * detected_steps / total_steps if total_steps else 0.0
    return 100.0 * accuracy / num_samples, 100.0 * iou / num_samples, recall


@torch.no_grad()
def evaluate_raw(
    dataset,
    model,
    gamma,
    drop_cost,
    keep_percentile,
    drop_cost_scale=1.0,
    use_unlabeled=True,
    occurrence_temporal_bias=0.0,
):
    accuracy = 0.0
    iou = 0.0
    detected_steps = 0
    total_steps = 0
    num_samples = 0
    model_device = next(model.parameters()).device
    iterator = tqdm(dataset, desc="Evaluating Raw", dynamic_ncols=True)
    for sample in iterator:
        if sample["num_steps"] < 1:
            continue
        work_sample = dict(sample)
        work_sample["frame_features"] = model.map_video(sample["frame_features"].to(model_device)).detach().cpu()
        work_sample["step_features"] = model.map_text(sample["step_features"].to(model_device)).detach().cpu()
        if drop_cost == "learn":
            distractor = model.compute_distractors(work_sample["step_features"].mean(0).to(model_device)).detach().cpu()
        else:
            distractor = None
        zx_costs, drop_costs, _ = compute_all_costs(
            work_sample,
            distractor,
            gamma,
            drop_cost_type=drop_cost,
            keep_percentile=keep_percentile,
            distinct_step_occurrences=True,
        )
        drop_costs = drop_costs * float(drop_cost_scale)
        zx_np = zx_costs.detach().cpu().numpy()
        zx_decode = _add_occurrence_temporal_bias(zx_np, occurrence_temporal_bias)
        assignment = drop_dtw(
            zx_decode,
            drop_costs.detach().cpu().numpy(),
            return_labels=True,
        ) - 1
        accuracy += framewise_accuracy(assignment, sample, use_unlabeled=use_unlabeled)
        iou += IoU(assignment, sample)
        sample_detected, sample_steps = step_point_recall(
            sample,
            work_sample["step_features"],
            work_sample["frame_features"],
        )
        detected_steps += sample_detected
        total_steps += sample_steps
        num_samples += 1
    recall = 100.0 * detected_steps / total_steps if total_steps else 0.0
    return 100.0 * accuracy / num_samples, 100.0 * iou / num_samples, recall


class PeriodicAndBestCheckpoint(pl.callbacks.Callback):
    def __init__(self, dirpath, save_every_n_epochs=1, log_file=None):
        super().__init__()
        self.dirpath = dirpath
        self.save_every_n_epochs = save_every_n_epochs
        self.log_file = log_file
        self.best_iou = None

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            os.makedirs(self.dirpath, exist_ok=True)
            record = pl_module.last_epoch_record or {"epoch": int(trainer.current_epoch)}
            if self.log_file is not None:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                with open(self.log_file, "a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

            epoch = int(trainer.current_epoch)
            if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                trainer.save_checkpoint(os.path.join(self.dirpath, f"weights-epoch={epoch:02d}.ckpt"))
                trainer.save_checkpoint(os.path.join(self.dirpath, "last.ckpt"))

            current_iou = record.get("tam_iou")
            if current_iou is not None and (self.best_iou is None or current_iou > self.best_iou):
                self.best_iou = float(current_iou)
                best_path = os.path.join(self.dirpath, "best.ckpt")
                trainer.save_checkpoint(best_path)
                best_record = dict(record)
                best_record["best_iou"] = self.best_iou
                with open(os.path.join(self.dirpath, "best_metrics.json"), "w") as handle:
                    json.dump(best_record, handle, indent=2, sort_keys=True)


def load_base_model_checkpoint(base_model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = {}
    valid_prefixes = ("model.", "base_model.")
    valid_roots = ("video_mapping.", "text_mapping.", "drop_mapping.")

    for key, value in source_state.items():
        candidates = [key]
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in valid_prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    candidates.append(stripped)
                    changed = True
                    break
        for candidate in candidates:
            if candidate.startswith(valid_roots):
                target_state[candidate] = value
                break

    if not target_state:
        example_keys = list(source_state.keys())[:10]
        raise RuntimeError(
            f"No EmbeddingsMapping weights found in {checkpoint_path}. "
            f"Example checkpoint keys: {example_keys}"
        )

    incompatible = base_model.load_state_dict(target_state, strict=False)
    print(
        "Loaded base Drop-DTW checkpoint from "
        f"{checkpoint_path}: {len(target_state)} tensors, "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )


def set_requires_grad(module, requires_grad):
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def memory_source_task_ids(memory_graph_path):
    payload = torch.load(memory_graph_path, map_location="cpu")
    summary = payload.get("summary", {})
    if "source_task_ids" in summary:
        return sorted(int(task_id) for task_id in summary["source_task_ids"])
    graphs = payload.get("graphs", {})
    return sorted(int(task_id) for task_id in graphs)


def filter_data_to_task_ids(data, task_ids):
    for split_name in ("train_dataset", "val_dataset", "test_dataset"):
        dataset = getattr(data, split_name, None)
        if dataset is None or not hasattr(dataset, "filter_task_ids"):
            continue
        before = len(dataset)
        dataset.filter_task_ids(task_ids)
        print(
            f"Filtered {split_name} to {len(task_ids)} memory tasks: "
            f"{before} -> {len(dataset)} samples"
        )


def main():
    if args.accumulate_grad_batches < 1:
        parser.error("--accumulate_grad_batches must be at least 1")
    if args.gpus < 1:
        parser.error("--gpus must be at least 1")
    if args.mem_loss_warmup_epochs < 0:
        parser.error("--mem_loss_warmup_epochs must be non-negative")
    if args.span_recall_loss_mult < 0:
        parser.error("--span_recall_loss_mult must be non-negative")
    if args.span_recall_topk < 1:
        parser.error("--span_recall_topk must be at least 1")
    data = DataModule(args.dataset, args.n_cls, args.batch_size)
    normalization_dataset = None if args.init_base_ckpt is not None else data.train_dataset
    base_model = EmbeddingsMapping(
        d=512,
        learnable_drop=(args.drop_cost == "learn"),
        video_layers=args.video_layers,
        text_layers=args.text_layers,
        normalization_dataset=normalization_dataset,
        batchnorm=args.batchnorm,
    )
    if args.pretrained_drop:
        from glob import glob

        weights_path = glob(os.path.join(WEIGHTS_PATH, args.name, "weights-epoch=*.ckpt"))[0]
        state_dict = {
            k[6:]: v
            for k, v in torch.load(weights_path, map_location=device)["state_dict"].items()
            if k.startswith("model.base_model.drop_mapping") or k.startswith("model.drop_mapping")
        }
        state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict, strict=False)

    if args.init_base_ckpt is not None:
        load_base_model_checkpoint(base_model, args.init_base_ckpt)
    else:
        print(
            "Base initialization: fresh Drop-DTW EmbeddingsMapping with dataset "
            "normalization; no trained Drop-DTW checkpoint loaded."
        )

    if args.freeze_base_model:
        set_requires_grad(base_model, False)
        print("Frozen Drop-DTW base model parameters; training TAM parameters only.")

    if args.memory_assignments is None:
        parser.error("--memory_assignments is required")
    if args.held_out_tasks_csv is None:
        parser.error("--held_out_tasks_csv is required")
    if args.retriever_topk < 1:
        parser.error("--retriever_topk must be at least 1")
    if args.memory_visual_init_weight < 0.0 or args.memory_time_init_weight < 0.0:
        parser.error("TAT modality initialization weights must be non-negative")
    if args.memory_visual_projection_residual_scale < 0.0:
        parser.error("--memory_visual_projection_residual_scale must be non-negative")
    if args.memory_temporal_hidden_dim < 1:
        parser.error("--memory_temporal_hidden_dim must be positive")
    if args.transition_log_epsilon <= 0:
        parser.error("--transition_log_epsilon must be positive")
    print(
        "TAT objective: L_mem + beta * L_T + L_U; "
        f"L_T/L_U use Drop-DTW baseline weights alignment={args.dropdtw_align_mult}, "
        f"clustering={args.clust_loss_mult}; drop_cost={args.drop_cost}."
    )
    print(
        "Memory node initialization keeps the semantic anchor and adds "
        "norm-calibrated visual/time residuals: "
        f"visual={args.memory_visual_init_weight}, "
        f"time={args.memory_time_init_weight}."
    )
    print(
        "Guided prediction P_u: score-only cosine / tau with "
        f"tau={args.guided_score_temperature}; decoder embeddings are unchanged."
    )
    print(
        "Decoder residual scale: "
        f"init={args.decoder_residual_scale_init}, "
        f"learnable_bounded={args.learnable_decoder_residual_scale}."
    )
    memory = TaskMemory(
        args.memory_graph,
        assignments_path=args.memory_assignments,
        held_out_tasks_csv=args.held_out_tasks_csv,
        d=512,
        visual_init_weight=args.memory_visual_init_weight,
        time_init_weight=args.memory_time_init_weight,
        multimodal_projection_mode=args.memory_projection_mode,
        calibrate_fusion_modalities=not args.disable_memory_fusion_calibration,
        visual_projection_residual_scale=args.memory_visual_projection_residual_scale,
        temporal_projection_hidden_dim=args.memory_temporal_hidden_dim,
        train_temporal_prototypes=not args.freeze_temporal_prototypes,
        train_temporal_edges=not args.freeze_temporal_edges,
        straight_through_retrieval=(args.retrieval_gradient == "straight_through"),
        retrieval_temperature=args.retrieval_temperature,
        decoder_prediction_head=args.decoder_prediction_head,
        decoder_residual_scale_init=args.decoder_residual_scale_init,
        learnable_decoder_residual_scale=args.learnable_decoder_residual_scale,
        guided_score_temperature=args.guided_score_temperature,
        guided_score_reference_gamma=(
            args.guided_score_reference_gamma
            if args.guided_score_reference_gamma > 0
            else None
        ),
        association_type=args.association_type,
        retriever_log_transition=args.retriever_log_transition,
        transition_log_epsilon=args.transition_log_epsilon,
    )
    model = TAMDropDTW(base_model, memory)
    train_module = TrainModule(model, data)

    checkpoint_root = args.checkpoint_root or os.path.join(PROJECT_PATH, "weights")
    checkpoint_dir = os.path.join(checkpoint_root, args.name)
    log_file = args.log_file or os.path.join(checkpoint_dir, "train_log.jsonl")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.resume_weights_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_weights_from_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = train_module.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded train module weights from {args.resume_weights_from_checkpoint}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)
    if args.reinit_memory_multimodal:
        train_module.model.memory.reinitialize_multimodal_node_paths(
            visual_weight=args.memory_visual_init_weight,
            time_weight=args.memory_time_init_weight,
        )
        print(
            "Reinitialized memory visual/time node paths with "
            f"visual_weight={args.memory_visual_init_weight}, "
            f"time_weight={args.memory_time_init_weight}"
        )

    with open(os.path.join(checkpoint_dir, "config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    checkpoint_callback = PeriodicAndBestCheckpoint(
        checkpoint_dir,
        save_every_n_epochs=args.save_every_n_epochs,
        log_file=log_file,
    )
    logger = pl.loggers.TensorBoardLogger("tb_logs", args.name)
    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy=(
            DDPStrategy(find_unused_parameters=True)
            if args.gpus > 1
            else None
        ),
        replace_sampler_ddp=False,
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
        logger=logger,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=(
            args.limit_train_batches if args.limit_train_batches > 0 else 1.0
        ),
    )
    trainer.fit(train_module, data)


if __name__ == "__main__":
    main()
