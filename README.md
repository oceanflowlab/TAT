# TAT

## 1. Install

Use Python 3.8 and install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are PyTorch, PyTorch Lightning, LMDB, PyArrow, NumPy, and
TQDM.

## 2. Prepare COIN Features

TAT uses S3D features for COIN.  The COIN feature package linked by the
original Drop-DTW project is no longer available, so the features need to be
extracted from the raw COIN videos.  This README does not cover how to obtain
the raw COIN videos; after the videos are prepared, use the feature extraction
pipeline from the original Drop-DTW project.

The Drop-DTW COIN preprocessing does the following:

1. Downloads the HowTo100M-pretrained S3D files:
   `s3d_howto100m.pth` and `s3d_dict.npy`.
2. Converts COIN videos into TFRecords.
3. Runs the S3D encoder on the TFRecords.
4. Packs the encoded features into LMDB files.
5. Creates `steps_info.pickle`, which stores step text descriptions and step
   text embeddings.

In the Drop-DTW project, this is handled by:

```bash
python video_encoding/setup_COIN.py
```

After extraction, put or symlink the resulting COIN feature directory to
`data/COIN/`, or set `DROPD_TW_COIN_PATH` to that directory before running this
code:

```bash
export DROPD_TW_COIN_PATH=/path/to/COIN
```

The COIN feature directory should have:

```text
COIN/
├── lmdb/
│   ├── <task_folder>/
│   │   ├── <task>_train.lmdb
│   │   ├── <task>_val.lmdb
│   │   └── <task>_test.lmdb
│   └── ...
└── steps_info.pickle
```

Each LMDB sample must contain:

```text
frames_features
steps_ids
steps_features
steps_starts
steps_ends
name
cls
cls_name
num_steps
num_subs
```

## 3. Train the Drop-DTW Starting Model

TAT is initialized from a trained Drop-DTW encoder.  This repository does not
include a Drop-DTW COIN checkpoint, so first train Drop-DTW on COIN with the
original Drop-DTW project.

In the Drop-DTW project, the basic COIN training command is:

```bash
python train.py --name drop_dtw_coin --keep_percentile 0.3
```

After training, use the selected Drop-DTW checkpoint as the initialization
checkpoint for TAT.  For the default commands below, place or symlink it here:

```text
weights/drop_dtw_coin/weights-epoch=13.ckpt
```

The checkpoint should contain the Drop-DTW embedding mapper weights:

```text
model.video_mapping.*
model.text_mapping.*
model.drop_mapping.*
```

## 4. Build TAT Memory

TAT memory is built from the COIN training split.  Run the following commands
from this repository root.

### 4.1 Collect Step Occurrences

This step uses the trained Drop-DTW model to align training videos and collect
one record per step occurrence.

```bash
python memory/build_step_occurrences.py \
  --dataset COIN \
  --split train \
  --ckpt weights/drop_dtw_coin/weights-epoch=13.ckpt \
  --output_dir outputs/coin_memory \
  --span_source pseudo \
  --drop_cost logit \
  --gamma 30 \
  --keep_percentile 0.3
```

Output:

```text
outputs/coin_memory/coin_train_step_occurrences_pseudo.pt
outputs/coin_memory/coin_train_step_occurrences_pseudo_summary.json
```

If oracle temporal spans are available for an analysis-only experiment, replace
`--span_source pseudo` with `--span_source gt`.

### 4.2 Build Memory Nodes

This step clusters repeated step occurrences within each task and builds memory
nodes with text, visual, and temporal prototypes.

```bash
python memory/build_task_memory_variants.py \
  --occurrences outputs/coin_memory/coin_train_step_occurrences_pseudo.pt \
  --steps_info "$DROPD_TW_COIN_PATH/steps_info.pickle" \
  --held_out_tasks_csv local_data/held_out_tasks.csv \
  --output outputs/coin_memory/task_memory_nodes.pt \
  --summary outputs/coin_memory/task_memory_nodes_summary.json \
  --node_budget 50 \
  --min_support 3 \
  --temporal_weight 0.5 \
  --phase_weight 0.25 \
  --prototype_trim_fraction 0.2 \
  --seed 10
```

`local_data/held_out_tasks.csv` lists tasks that should not be used to build
memory.  It should contain:

```text
task_id,nearest_train_task_id
```

`task_id` is used to exclude held-out tasks during memory construction.
`nearest_train_task_id` is used at evaluation time to choose the training task
memory for a held-out task.

### 4.3 Build the Memory Graph

This step adds semantic and temporal edges between memory nodes.

```bash
python memory/build_task_memory_graph.py \
  --occurrences outputs/coin_memory/coin_train_step_occurrences_pseudo.pt \
  --variants outputs/coin_memory/task_memory_nodes.pt \
  --output outputs/coin_memory/task_memory_graph.pt \
  --summary outputs/coin_memory/task_memory_graph_summary.json \
  --semantic_topk_steps 3 \
  --min_temporal_support 2
```

The two files needed for training are:

```text
outputs/coin_memory/task_memory_graph.pt
outputs/coin_memory/task_memory_nodes.pt
```

## 5. Train TAT

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --name tat_coin \
  --checkpoint_root weights \
  --memory_graph outputs/coin_memory/task_memory_graph.pt \
  --memory_assignments outputs/coin_memory/task_memory_nodes.pt \
  --held_out_tasks_csv local_data/held_out_tasks.csv \
  --init_base_ckpt weights/drop_dtw_coin/weights-epoch=13.ckpt
```

Training writes checkpoints and logs to:

```text
weights/tat_coin/
├── config.json
├── train_log.jsonl
├── weights-epoch=XX.ckpt
├── last.ckpt
├── best.ckpt
└── best_metrics.json
```

## 6. Evaluate

Evaluate the trained model:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --run tat_coin \
  --ckpt weights/tat_coin/best.ckpt \
  --split test
```

The output JSON contains accuracy, recall, and IoU.
