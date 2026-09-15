import argparse
import json
from pathlib import Path

import torch

from datasets.data_module import DataModule
from engine import EVAL_DROP_COST_SCALE, build_model, evaluate, load_tat_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TAT on COIN")
    parser.add_argument("--checkpoint", default="weights/tat_coin/best.ckpt")
    parser.add_argument("--memory_graph", default="outputs/coin_memory/task_memory_graph.pt")
    parser.add_argument("--memory_assignments", default="outputs/coin_memory/task_memory_nodes.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--drop_cost_scale", type=float, default=EVAL_DROP_COST_SCALE)
    parser.add_argument("--output", default="results/tat_coin_val.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.drop_cost_scale <= 0:
        raise ValueError("--drop_cost_scale must be positive")
    data = DataModule(batch_size=16, videos_per_task=2)
    model = build_model(args.memory_graph, args.memory_assignments)
    load_tat_checkpoint(model, args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    dataset = data.val_dataset if args.split == "val" else data.test_dataset
    accuracy, iou, recall = evaluate(
        dataset, model, drop_cost_scale=args.drop_cost_scale
    )
    result = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "drop_cost_scale": args.drop_cost_scale,
        "accuracy": accuracy,
        "recall": recall,
        "iou": iou,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
