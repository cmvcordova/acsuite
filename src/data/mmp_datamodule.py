import os, sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch, torch-geometric
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from MoleculeACE import Data
import molfeat

class SiameseACDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.mols = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        x = self.mols['smiles'][idx]
        if self.transform:
            x = self.transform(x)
        
        y = self.mols['acd'][idx]
        y = torch.tensor(y, dtype=torch.float32)
        return x1, x2, y

class mmp_datamodule(pl.LightningDataModule):
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
    
    def __init__(self, csv_file, root_dir, batch_size, num_workers, shuffle=True):
        super().__init__()
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def prepare_data(self) -> None:
        print('Preparing data...')
        

    def setup(self, stage=None):
        self.ac_dataset = ACDataset(self.csv_file, self.root_dir, transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.ac_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.ac_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def test_dataloader(self):
        return DataLoader(self.ac_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

if __name__ == '__main__':
    _ = ACDataloader()



