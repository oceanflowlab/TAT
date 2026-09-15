# TAT

This repository contains the training and evaluation code for TAT on COIN.

## Repository layout

```text
TAT/
├── README.md
├── requirements.txt
├── train.py
├── eval.py
├── engine.py
├── datasets/
├── models/
├── memory/
├── dp/
└── utils/
```

Run every command below from the repository root.

## 1. Environment

Create a Python 3.8 environment and install the runtime dependencies:

```bash
conda create -n tat python=3.8 -y
conda activate tat
pip install -r requirements.txt
```

Feature extraction additionally needs the packages required by the official
[Drop-DTW repository](https://github.com/SamsungLabs/Drop-DTW), including
TensorFlow, pandas, OpenCV, and the TFRecord utilities used there.

The LMDB loader opens many files. Raise the process file limit before building
memory, training, or evaluating:

```bash
ulimit -n 5000
```

## 2. COIN features

The pre-extracted COIN features formerly linked by Drop-DTW are no longer
available. Extract the S3D video and text features from the COIN videos using
the official Drop-DTW preprocessing code. Obtaining the COIN videos themselves
is outside the scope of this repository.

Clone Drop-DTW next to this repository:

```bash
git clone https://github.com/SamsungLabs/Drop-DTW.git ../Drop-DTW
```

Download the HowTo100M-pretrained S3D checkpoint and text dictionary:

```bash
mkdir -p ../Drop-DTW/video_encoding/model_weights
wget -O ../Drop-DTW/video_encoding/model_weights/s3d_howto100m.pth \
  https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget -O ../Drop-DTW/video_encoding/model_weights/s3d_dict.npy \
  https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
```

In the Drop-DTW checkout, set `COIN_PATH` in `paths.py` to the COIN data
directory. That directory must contain `COIN.json`, `taxonomy_step.csv`, and
the prepared videos in the layout expected by
`video_encoding/InstVids2TFRecord_COIN.py`.

Convert the videos to TFRecords and encode them into LMDB files:

```bash
cd ../Drop-DTW
python video_encoding/InstVids2TFRecord_COIN.py --mode=train
python video_encoding/InstVids2TFRecord_COIN.py --mode=val
python video_encoding/encode_lmdb.py \
  --source data/COIN/videos_tfrecords \
  --dest data/COIN/lmdb \
  --dataset COIN
cd ../TAT
```

`encode_lmdb.py` also creates `steps_info.pickle`, containing the COIN step
descriptions and S3D text embeddings. Place the completed feature directory at
`data/COIN`:

```text
data/COIN/
├── lmdb/
│   ├── <task>/
│   │   ├── <task>_train.lmdb
│   │   └── <task>_val.lmdb
│   └── ...
└── steps_info.pickle
```

For a different relative location, set `TAT_COIN_PATH` before running TAT.

## 3. Drop-DTW initialization

TAT starts from a trained Drop-DTW video/text mapping. Follow the training
instructions in the official Drop-DTW repository and train its COIN model with
the 0.3 percentile drop cost:

```bash
cd ../Drop-DTW
python train.py --name=drop_dtw_coin --keep_percentile=0.3
cd ../TAT
```

Select the Drop-DTW checkpoint using its validation results and place it at:

```text
weights/drop_dtw_coin/best.ckpt
```

## 4. Build task memory

Memory is constructed only from pseudo alignments produced by the trained
Drop-DTW model. The three commands below use the released TAT configuration.

Collect the training step occurrences:

```bash
python memory/build_step_occurrences.py
```

Cluster occurrences into multimodal memory nodes:

```bash
python memory/build_task_memory_variants.py
```

Build semantic and temporal graph edges:

```bash
python memory/build_task_memory_graph.py
```

The resulting files are:

```text
outputs/coin_memory/
├── coin_train_step_occurrences_pseudo.pt
├── coin_train_step_occurrences_pseudo_summary.json
├── task_memory_nodes.pt
├── task_memory_nodes_summary.json
├── task_memory_graph.pt
└── task_memory_graph_summary.json
```

## 5. Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

The run is written to `weights/tat_coin/`. The directory contains the resolved
configuration, epoch metrics, the latest checkpoint, and the checkpoint with
the best validation IoU:

```text
weights/tat_coin/
├── config.json
├── train_log.jsonl
├── last.ckpt
├── best.ckpt
└── best_metrics.json
```

To resume an interrupted run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --resume weights/tat_coin/last.ckpt
```

## 6. Evaluate

Evaluate the best checkpoint on the COIN validation split:

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --checkpoint weights/tat_coin/best.ckpt \
  --split val
```

The metrics are printed and saved to `results/tat_coin_val.json`.
