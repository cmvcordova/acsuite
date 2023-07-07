# standard python/pytorch imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, TypedDict, Union, Literal

## Molfeat related imports
import datamol as dm
from molfeat.calc import FP_FUNCS, FPCalculator
from molfeat.trans.concat import FeatConcat
from molfeat.trans import MoleculeTransformer
## MoleculeACE imports
from MoleculeACE import Data as moleculeace_Data


class MoleculeACEDataset(Dataset):
    """
    Wrapper that creates a pytorch dataloader compliant dataset for molecular property prediction data 
    from the datasets as provided by MoleculeACE:
    van Tilborg et al 2022.
    https://github.com/molML/MoleculeACE
    Supports molfeat featurizer schemes and classification/regression tasks.

    Args:
        dataset_name: Name of the dataset to be used e.g. "CHEMBL234"
        data_split: Data split to be used, either "train" or "test"
        task: Task to be performed, either "regression" or "classification"
        molfeat_featurizer: MolFeat featurizer
    """
    def __init__(self,
    dataset_name: str = None,
    data_split: str = 'train',
    task: Literal['regression', 'classification'] = 'classification',
    molfeat_featurizer = MoleculeTransformer(
        FPCalculator('ecfp', 
        length = 2048,
        radius = 4)
        #, dtype = torch.float32 didn't work
        )
    ):
        self.dataset = moleculeace_Data(dataset_name)
        self.molfeat_featurizer = molfeat_featurizer

        if task == 'regression':
            if data_split == 'train':
                self.x = molfeat_featurizer(self.dataset.smiles_train)
                self.y = self.dataset.y_train
            elif data_split == 'test':
                self.x = molfeat_featurizer(self.dataset.smiles_test)
                self.y = self.dataset.y_test

        elif task == 'classification':
            if data_split == 'train':
                self.x = molfeat_featurizer(self.dataset.smiles_train)
                self.y = self.dataset.cliff_mols_train
            elif data_split == 'test':
                self.x = molfeat_featurizer(self.dataset.smiles_test)
                self.y = self.dataset.cliff_mols_test

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        molecule = torch.as_tensor(self.x[idx], dtype = torch.float32)
        label = torch.tensor(self.y[idx], dtype = torch.long)
        return molecule, label
    
