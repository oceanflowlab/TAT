import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from utils.paths import COIN_PATH


def build_train_argv(config):
    argv = ["engine.py"]
    true_flags = {
        "freeze_base_model",
        "decode_frames",
        "disable_message_passing",
        "normalize_enhanced",
        "pretrained_drop",
        "memory_as_step",
        "freeze_temporal_edges",
        "freeze_temporal_prototypes",
        "disable_light_decoder",
        "full_dropdtw_loss",
        "learnable_decoder_residual_scale",
        "skip_epoch_eval",
        "retriever_log_transition",
    }
    for key, value in config.items():
        if isinstance(value, bool):
            if value and key in true_flags:
                argv.append("--" + key)
            elif not value and key == "freeze_memory_prototypes":
                argv.append("--train_memory_prototypes")
            elif not value and key == "residual_decoder":
                argv.append("--non_residual_decoder")
        elif value is not None:
            argv.extend(["--" + key, str(value)])
    return argv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--manifest_jsonl", default=None)
    parser.add_argument("--steps_info", default=None)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif args.run is not None:
        run_dir = root / "weights" / args.run
    else:
        parser.error("one of --run or --run_dir is required")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    sys.argv = build_train_argv(config)
    os.environ.setdefault("TAM_EVAL_PROGRESS", "1")

    import engine
    from datasets.data_module import DataModule
    from datasets.data_utils import Time2FrameNumber, dict2tensor
    from models.nets import EmbeddingsMapping
    from models.task_memory import TaskMemory

    class RouteManifestDataset(torch.utils.data.Dataset):
        def __init__(self, manifest_jsonl, steps_info):
            self.rows = []
            with open(manifest_jsonl, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self.rows.append(json.loads(line))
            with open(steps_info, "rb") as handle:
                info = pickle.load(handle)
            self.step_embeddings = info["steps_to_embeddings"]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            frame_features = np.load(row["feature_path"]).astype(np.float32)
            step_ids = np.array(
                [int(item["step_id"]) for item in row["annotations"]],
                dtype=np.int64,
            )
            step_features = np.concatenate(
                [self.step_embeddings[int(step_id)] for step_id in step_ids],
                axis=0,
            ).astype(np.float32)
            starts_sec = np.array(
                [float(item["start_sec"]) for item in row["annotations"]],
                dtype=np.float32,
            )
            ends_sec = np.array(
                [float(item["end_sec"]) for item in row["annotations"]],
                dtype=np.float32,
            )
            starts = np.array(
                [Time2FrameNumber(float(value), 10) // 32 for value in starts_sec],
                dtype=np.int64,
            )
            ends = np.array(
                [Time2FrameNumber(float(value), 10) // 32 for value in ends_sec],
                dtype=np.int64,
            )
            last_frame = max(int(frame_features.shape[0]) - 1, 0)
            starts = np.clip(starts, 0, last_frame)
            ends = np.clip(ends, 0, last_frame)
            sample = {
                "name": row["video_id"],
                "cls": np.array(int(row["task_id"]), dtype=np.int64),
                "cls_name": row["task_name"],
                "num_steps": np.array(len(step_ids), dtype=np.int64),
                "num_subs": np.array(len(step_ids), dtype=np.int64),
                "frame_features": frame_features,
                "num_frames": np.array(frame_features.shape[0], dtype=np.int64),
                "step_ids": step_ids,
                "steps": [item["step_text"] for item in row["annotations"]],
                "step_features": step_features,
                "step_starts_sec": starts_sec,
                "step_ends_sec": ends_sec,
                "step_starts": starts,
                "step_ends": ends,
            }
            return dict2tensor(sample)

    data = DataModule(engine.args.dataset, engine.args.n_cls, engine.args.batch_size)
    base = EmbeddingsMapping(
        d=512,
        learnable_drop=(engine.args.drop_cost == "learn"),
        video_layers=engine.args.video_layers,
        text_layers=engine.args.text_layers,
        normalization_dataset=None,
        batchnorm=engine.args.batchnorm,
    )
    memory = TaskMemory(
        engine.args.memory_graph,
        assignments_path=engine.args.memory_assignments,
        held_out_tasks_csv=engine.args.held_out_tasks_csv,
        d=512,
        visual_init_weight=engine.args.memory_visual_init_weight,
        time_init_weight=engine.args.memory_time_init_weight,
        multimodal_projection_mode=engine.args.memory_projection_mode,
        calibrate_fusion_modalities=(
            not engine.args.disable_memory_fusion_calibration
        ),
        visual_projection_residual_scale=(
            engine.args.memory_visual_projection_residual_scale
        ),
        temporal_projection_hidden_dim=engine.args.memory_temporal_hidden_dim,
        train_temporal_prototypes=not engine.args.freeze_temporal_prototypes,
        train_temporal_edges=not engine.args.freeze_temporal_edges,
        straight_through_retrieval=(
            engine.args.retrieval_gradient == "straight_through"
        ),
        retrieval_temperature=engine.args.retrieval_temperature,
        decoder_prediction_head=engine.args.decoder_prediction_head,
        decoder_residual_scale_init=engine.args.decoder_residual_scale_init,
        learnable_decoder_residual_scale=(
            engine.args.learnable_decoder_residual_scale
        ),
        guided_score_temperature=engine.args.guided_score_temperature,
        guided_score_reference_gamma=(
            engine.args.guided_score_reference_gamma
            if engine.args.guided_score_reference_gamma > 0
            else None
        ),
        association_type=engine.args.association_type,
        retriever_log_transition=engine.args.retriever_log_transition,
        transition_log_epsilon=engine.args.transition_log_epsilon,
    )
    model = engine.TAMDropDTW(base, memory)

    checkpoint_path = Path(args.ckpt)
    checkpoint = torch.load(checkpoint_path, map_location=engine.device)
    state = {
        key[len("model.") :]: value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(engine.device)
    model.eval()

    if args.manifest_jsonl is not None:
        steps_info = args.steps_info or str(Path(COIN_PATH) / "steps_info.pickle")
        dataset = RouteManifestDataset(args.manifest_jsonl, steps_info)
    else:
        dataset = data.val_dataset if args.split == "val" else data.test_dataset
    tam_acc, tam_iou, tam_recall = engine.evaluate_tam(
        dataset,
        model,
        gamma=engine.args.step_xz_gamma,
        drop_cost=engine.args.drop_cost,
        keep_percentile=engine.args.keep_percentile,
        drop_cost_scale=engine.args.drop_cost_scale,
        use_unlabeled=True,
    )
    raw_acc, raw_iou, raw_recall = engine.evaluate_raw(
        dataset,
        model,
        gamma=engine.args.step_xz_gamma,
        drop_cost=engine.args.drop_cost,
        keep_percentile=engine.args.keep_percentile,
        drop_cost_scale=engine.args.drop_cost_scale,
        use_unlabeled=True,
    )
    record = {
        "run": args.run,
        "run_dir": str(run_dir),
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "tam_accuracy": tam_acc,
        "tam_recall": tam_recall,
        "tam_iou": tam_iou,
        "raw_accuracy": raw_acc,
        "raw_recall": raw_recall,
        "raw_iou": raw_iou,
    }
    text = json.dumps(record, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n")


if __name__ == "__main__":
    main()
