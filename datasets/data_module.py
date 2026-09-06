import os
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datasets.loader import LMDB_Folder_Dataset
from datasets.batching import BatchIdxSampler_Class, flatten_batch
from datasets.data_utils import dict2tensor
from utils.paths import CT_PATH, COIN_PATH, YC_PATH, ARA_PATH, MR_PATH


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, n_cls, batch_size):
        super().__init__()
        if dataset_name == "COIN":
            folder = COIN_PATH
        elif dataset_name == "YouCook2":
            folder = YC_PATH
        elif dataset_name == "CrossTask":
            folder = CT_PATH
        elif dataset_name == "ARA":
            folder = ARA_PATH
        elif dataset_name == "MR":
            folder = MR_PATH
        else:
            raise f"No such dataset {dataset_name}"
        
        self.lmdb_path = os.path.join(folder, "lmdb")
        self.n_cls = n_cls
        self.batch_size = batch_size

        self.train_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="train", transform=dict2tensor)
        self.val_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="val", transform=dict2tensor)
        self.test_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="test", transform=dict2tensor)
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
