
# standard python/pytorch imports
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, TypedDict, Union

# util funcitons
from src.utils.data import read_ACNet_single_line_JSON_file

## Molfeat related imports
import datamol as dm
from molfeat.calc import FP_FUNCS, FPCalculator
from molfeat.trans.concat import FeatConcat
from molfeat.trans import MoleculeTransformer

class MMPDataset(Dataset):
    """
    Create a dataset for MMPs from a JSON file as provided after generating ACNet datasets.
    
    Args:

    """
    def __init__(self,
        mmp_df: pd.DataFrame,
        smiles_one: str,
        smiles_two: str,
        label: str,
        target: str,
        molfeat_featurizer = MoleculeTransformer(
            FPCalculator('ecfp', 
            length = 1024,
            radius = 2)
        ),
        output_type: str = 'concat',
        target_dict: Optional[TypedDict] = None,
    ):
        self.smiles_one = mmp_df[smiles_one].values
        self.smiles_two = mmp_df[smiles_two].values
        self.label = mmp_df[label].values
        self.target = mmp_df[target].values
        self.output_type = output_type
        self.molfeat_featurizer = molfeat_featurizer
        self.target_dict = target_dict

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        molecule_one = self.molfeat_featurizer(self.smiles_one[idx])
        molecule_two = self.molfeat_featurizer(self.smiles_two[idx])
        target = self.target[idx]
        label = self.label[idx]
        if self.output_type == 'pair':
            return molecule_one, molecule_two, target, label
        elif self.output_type == 'concat':
            return np.concatenate((molecule_one, molecule_two), axis = None), target, label
