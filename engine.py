import argparse
import json
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm

from datasets.batching import unflatten_batch
from datasets.data_module import DataModule
from dp.dp_utils import compute_all_costs
from dp.exact_dp import crosstask_dp, drop_dtw
from models.losses import compute_alignment_loss, compute_clustering_loss
from models.nets import EmbeddingsMapping
from models.task_memory import TaskMemory
from utils.metrics import CLIP_SECONDS, IoU, framewise_accuracy


TRAIN_DROP_COST_SCALE = 1.5
EVAL_DROP_COST_SCALE = 0.7
ALIGNMENT_WEIGHT = 2.5
CLUSTERING_WEIGHT = 4.0
STEP_GAMMA = 30.0
KEEP_PERCENTILE = 0.3


class TATModel(torch.nn.Module):
    def __init__(self, encoder, memory):
        super().__init__()
        self.encoder = encoder
        self.memory = memory

    def map_video(self, features):
        return self.encoder.map_video(features)

    def map_text(self, features):
        return self.encoder.map_text(features)

    def enhance_sample(self, sample):
        result = self.memory.associate_retrieved_route(
            sample,
            self.map_text,
            self.map_video,
        )
        if result is None:
            task_id = int(sample["cls"].detach().cpu())
            raise RuntimeError("No memory graph was found for task {}".format(task_id))
        return result["sample"]

    def enhance_samples(self, samples):
        return [self.enhance_sample(sample) for sample in samples]

    def memory_constraint_loss(self, samples):
        return self.memory.memory_constraint_loss(
            samples,
            self.map_text,
            self.map_video,
        )


def build_model(memory_graph, memory_assignments):
    encoder = EmbeddingsMapping(d=512)
    memory = TaskMemory(memory_graph, memory_assignments, d=512)
    return TATModel(encoder, memory)


def load_encoder_checkpoint(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    target_state = {}
    for key, value in source_state.items():
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in ("model.", "base_model.", "encoder."):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
                    break
        if stripped.startswith(("video_mapping.", "text_mapping.")):
            target_state[stripped] = value
    if not target_state:
        raise RuntimeError(
            "The checkpoint does not contain Drop-DTW video/text mapper weights: "
            + str(checkpoint_path)
        )
    incompatible = encoder.load_state_dict(target_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected encoder keys: {}".format(incompatible.unexpected_keys)
        )
    print("Loaded {} encoder tensors from {}".format(len(target_state), checkpoint_path))


def load_tat_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("state_dict", checkpoint)
    state = {}
    legacy_memory_prefix = "".join(("r", "aw_"))
    for key, value in source_state.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if key.startswith("base_model."):
            key = "encoder." + key[len("base_model.") :]
        if key.startswith("memory." + legacy_memory_prefix + "text_"):
            key = "memory.source_text_" + key.split("_")[-1]
        elif key.startswith("memory." + legacy_memory_prefix + "visual_"):
            key = "memory.source_visual_" + key.split("_")[-1]
        state[key] = value
    unused_checkpoint_keys = [
        key for key in state if ".frame_norm." in key
    ]
    for key in unused_checkpoint_keys:
        state.pop(key)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible: missing={}, unexpected={}".format(
                list(incompatible.missing_keys), list(incompatible.unexpected_keys)
            )
        )


class TrainModule(pl.LightningModule):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.epoch_sums = {}
        self.epoch_batches = 0
        self.last_epoch_record = None

    def configure_optimizers(self):
        encoder_parameters = [
            parameter
            for parameter in self.model.encoder.parameters()
            if parameter.requires_grad
        ]
        memory_parameters = []
        frame_to_route_parameters = []
        for name, parameter in self.model.memory.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".frame_to_route." in name:
                frame_to_route_parameters.append(parameter)
            else:
                memory_parameters.append(parameter)
        optimizer = torch.optim.Adam(
            [
                {
                    "params": encoder_parameters,
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                    "name": "encoder",
                },
                {
                    "params": memory_parameters,
                    "lr": 3e-5,
                    "weight_decay": 1e-4,
                    "name": "memory",
                },
                {
                    "params": frame_to_route_parameters,
                    "lr": 3e-5,
                    "weight_decay": 0.0,
                    "name": "frame_to_route",
                },
            ]
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 40, 60, 80], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, flat_batch, batch_idx):
        flat_batch["frame_features"] = self.model.map_video(
            flat_batch["frame_features"]
        )
        flat_batch["step_features"] = self.model.map_text(
            flat_batch["step_features"]
        )
        source_samples = unflatten_batch(flat_batch)
        samples = self.model.enhance_samples(source_samples)

        memory_losses = self.model.memory_constraint_loss(source_samples)
        alignment_loss = compute_alignment_loss(
            samples,
            drop_cost_scale=TRAIN_DROP_COST_SCALE,
        )
        clustering_loss = compute_clustering_loss(samples)
        task_loss = ALIGNMENT_WEIGHT * alignment_loss + CLUSTERING_WEIGHT * clustering_loss
        total_loss = memory_losses["loss"] + task_loss
        values = {
            "memory_loss": memory_losses["loss"],
            "semantic_mcl": memory_losses["semantic"],
            "temporal_mcl": memory_losses["temporal"],
            "alignment_loss": alignment_loss,
            "clustering_loss": clustering_loss,
            "task_loss": task_loss,
            "total_loss": total_loss,
        }
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite loss at epoch {}, batch {}: {}".format(
                    int(self.current_epoch),
                    batch_idx,
                    {key: float(value.detach().cpu()) for key, value in values.items()},
                )
            )
        for name, value in values.items():
            self.log("train/" + name, value)
            self.epoch_sums[name] = self.epoch_sums.get(name, 0.0) + float(
                value.detach().cpu()
            )
        self.epoch_batches += 1
        if batch_idx % 25 == 0 and self.trainer.is_global_zero:
            payload = {
                key: round(float(value.detach().cpu()), 6)
                for key, value in values.items()
            }
            print(
                "TRAIN epoch={} batch={}: {}".format(
                    int(self.current_epoch), batch_idx, json.dumps(payload, sort_keys=True)
                )
            )
        return total_loss

    def training_epoch_end(self, outputs):
        self.model.eval()
        accuracy, iou, recall = evaluate(
            self.data.val_dataset,
            self.model,
            drop_cost_scale=TRAIN_DROP_COST_SCALE,
        )
        self.log("metrics/accuracy", accuracy)
        self.log("metrics/recall", recall)
        self.log("metrics/iou", iou)
        record = {
            "epoch": int(self.current_epoch),
            "accuracy": float(accuracy),
            "recall": float(recall),
            "iou": float(iou),
        }
        if self.epoch_batches:
            record.update(
                {
                    "train_" + key: value / self.epoch_batches
                    for key, value in sorted(self.epoch_sums.items())
                }
            )
        record["learning_rates"] = {
            group.get("name", str(index)): float(group["lr"])
            for index, group in enumerate(self.trainer.optimizers[0].param_groups)
        }
        self.last_epoch_record = record
        self.epoch_sums = {}
        self.epoch_batches = 0
        print(
            "Validation epoch {}: Accuracy {:.3f}, Recall {:.3f}, IoU {:.3f}".format(
                int(self.current_epoch), accuracy, recall, iou
            )
        )
        self.model.train()


def step_point_recall(sample, step_features, frame_features, pairwise_scores):
    if pairwise_scores is None:
        pairwise_scores = step_features @ frame_features.T
    similarity = pairwise_scores.detach().cpu().numpy()
    assignment = crosstask_dp(-similarity.T).argmax(0)
    detected = 0
    count = int(sample["num_steps"].detach().cpu())
    for step_index in range(count):
        start = float(sample["step_starts_sec"][step_index].detach().cpu())
        end = float(sample["step_ends_sec"][step_index].detach().cpu())
        inferred_time = (int(assignment[step_index]) + 0.5) * CLIP_SECONDS
        detected += int(start <= inferred_time <= end)
    return detected, count


@torch.no_grad()
def evaluate(dataset, model, drop_cost_scale=EVAL_DROP_COST_SCALE):
    accuracy = 0.0
    iou = 0.0
    detected_steps = 0
    total_steps = 0
    sample_count = 0
    model_device = next(model.parameters()).device
    for sample in tqdm(dataset, desc="Evaluating TAT", dynamic_ncols=True):
        if int(sample["num_steps"]) < 1:
            continue
        work_sample = dict(sample)
        work_sample["frame_features"] = model.map_video(
            sample["frame_features"].to(model_device)
        )
        work_sample["step_features"] = model.map_text(
            sample["step_features"].to(model_device)
        )
        work_sample = {
            key: value.to(model_device) if torch.is_tensor(value) else value
            for key, value in work_sample.items()
        }
        work_sample = model.enhance_sample(work_sample)
        cpu_sample = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in work_sample.items()
        }
        match_costs, drop_costs, _ = compute_all_costs(
            cpu_sample,
            STEP_GAMMA,
            keep_percentile=KEEP_PERCENTILE,
            distinct_step_occurrences=True,
        )
        assignment = drop_dtw(
            match_costs.numpy(),
            (drop_costs * float(drop_cost_scale)).numpy(),
            return_labels=True,
        ) - 1
        accuracy += framewise_accuracy(assignment, sample, use_unlabeled=True)
        iou += IoU(assignment, sample)
        detected, count = step_point_recall(
            sample,
            cpu_sample["step_features"],
            cpu_sample["frame_features"],
            cpu_sample.get("pairwise_scores"),
        )
        detected_steps += detected
        total_steps += count
        sample_count += 1
    if sample_count == 0:
        raise RuntimeError("The evaluation split contains no usable samples")
    return (
        100.0 * accuracy / sample_count,
        100.0 * iou / sample_count,
        100.0 * detected_steps / total_steps if total_steps else 0.0,
    )


class BestCheckpoint(pl.callbacks.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.best_iou = None
        metrics_path = self.output_dir / "best_metrics.json"
        if metrics_path.exists():
            self.best_iou = float(json.loads(metrics_path.read_text())["iou"])

    def on_train_epoch_end(self, trainer, module):
        if not trainer.is_global_zero or module.last_epoch_record is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = module.last_epoch_record
        with (self.output_dir / "train_log.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        trainer.save_checkpoint(str(self.output_dir / "last.ckpt"))
        if self.best_iou is None or record["iou"] > self.best_iou:
            self.best_iou = record["iou"]
            trainer.save_checkpoint(str(self.output_dir / "best.ckpt"))
            best_record = dict(record)
            best_record["checkpoint"] = "best.ckpt"
            with (self.output_dir / "best_metrics.json").open("w") as handle:
                json.dump(best_record, handle, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train TAT on COIN")
    parser.add_argument("--name", default="tat_coin")
    parser.add_argument("--memory_graph", default="outputs/coin_memory/task_memory_graph.pt")
    parser.add_argument("--memory_assignments", default="outputs/coin_memory/task_memory_nodes.pt")
    parser.add_argument("--init_checkpoint", default="weights/drop_dtw_coin/best.ckpt")
    parser.add_argument("--output_root", default="weights")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpus < 1:
        raise ValueError("--gpus must be at least 1")
    random.seed(40)
    np.random.seed(40)
    torch.manual_seed(40)
    data = DataModule(batch_size=args.batch_size, videos_per_task=2)
    model = build_model(args.memory_graph, args.memory_assignments)
    load_encoder_checkpoint(model.encoder, args.init_checkpoint)
    module = TrainModule(model, data)

    output_dir = Path(args.output_root) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": args.name,
        "memory_graph": args.memory_graph,
        "memory_assignments": args.memory_assignments,
        "init_checkpoint": args.init_checkpoint,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": 40,
        "training_drop_cost_scale": TRAIN_DROP_COST_SCALE,
        "evaluation_drop_cost_scale": EVAL_DROP_COST_SCALE,
    }
    with (output_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    trainer = pl.Trainer(
        gpus=args.gpus,
        strategy=DDPStrategy(find_unused_parameters=True) if args.gpus > 1 else None,
        replace_sampler_ddp=False,
        callbacks=[BestCheckpoint(output_dir)],
        max_epochs=args.epochs,
        logger=pl.loggers.TensorBoardLogger("logs", args.name),
        resume_from_checkpoint=args.resume,
        gradient_clip_val=1.0,
    )
    trainer.fit(module, data)


if __name__ == "__main__":
    main()
