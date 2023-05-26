import os, sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch, torch_geometric
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from src.utils.data import read_ACNet_single_line_JSON_file
from src.data.mmp_dataset import MMPDataset

class MMPDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for MMP AC datasets.
    Args:
        data_path (str): Path to the data
        sequence_transform (callable): Transformation for the sequence
        cell_type_transform (callable): Transformation for the cell type
        batch_size (int): Batch size
        num_workers (int): Number of workers
    """

    df_train = None
    df_val = None
    df_test = None

    train_dataset: Dataset = None
    val_dataset: Dataset = None
    test_dataset: Dataset = None
    
    def __init__(
        ## add type hints here
        self, 
        file_name,
        data_dir,
        train_val_test_split,
        batch_size, 
        num_workers,
        pin_memory, 
        shuffle
    ):
        super().__init__()

        self.file_name = file_name
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    def prepare_data(self) -> None:
        print('Preprocessing data...')
        ## add additional split info here
        self.mmp_df = read_ACNet_single_line_JSON_file(self.data_dir + self.file_name)
        
    def setup(self, seed: int = 42, stage=None):
        print('Setting up data...')
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            """ may have to split into following convention, keep reg for now
            trainset = MMPDataset(self.mmp_df)
            testset = MMPDataset(self.mmp_df)
            dataset = ConcatDataset(datasets=[trainset, testset]
            """
            dataset = MMPDataset(self.mmp_df, 'SMILES1', 'SMILES2', 'Value', 'Target')
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(seed),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle
        )
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

if __name__ == '__main__':
    _ = MMPDataModule()



