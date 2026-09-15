import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datasets.loader import LMDB_Folder_Dataset
from datasets.batching import BatchIdxSampler_Class, flatten_batch
from datasets.data_utils import dict2tensor
from utils.paths import COIN_PATH


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=16, videos_per_task=2):
        super().__init__()
        self.lmdb_path = os.path.join(COIN_PATH, "lmdb")
        if not os.path.isdir(self.lmdb_path):
            raise FileNotFoundError(
                "COIN LMDB directory was not found at {}. "
                "Set TAT_COIN_PATH to the extracted COIN feature directory "
                "or place the features under data/COIN/.".format(self.lmdb_path)
            )
        self.n_cls = videos_per_task
        self.batch_size = batch_size

        self.train_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="train", transform=dict2tensor)
        self.val_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="val", transform=dict2tensor)
        self.test_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="test", transform=dict2tensor)
        if len(self.train_dataset) == 0:
            raise RuntimeError(
                "No COIN training samples were found under {}. "
                "Please check that the directory contains per-task train LMDB files.".format(self.lmdb_path)
            )
        print(len(self.train_dataset), len(self.val_dataset))

    def train_dataloader(self):
        batch_idx_sampler = BatchIdxSampler_Class(self.train_dataset, self.n_cls, self.batch_size)
        train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_idx_sampler,
            collate_fn=flatten_batch,
            num_workers=32,
            shuffle=False,
            drop_last=False,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, collate_fn=flatten_batch, num_workers=32)
        return val_loader

    def test_dataloader(self):
        val_loader = DataLoader(self.test_dataset, collate_fn=flatten_batch, num_workers=32)
        return val_loader
