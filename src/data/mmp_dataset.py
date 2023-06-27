
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
        mmp_df: MMP dataframe, instantiated from a JSON file as provided after generating ACNet datasets.
        smiles_one: Name of the column containing the first SMILES string
        smiles_two: Name of the column containing the second SMILES string
        label: Name of the column containing the label, corresponds to a binary label in the ACNet dataset 
            that indicates whether the two molecules represent an activity cliff or not
        target: Name of the column containing the target value, corresponds to the protein target value in the ACNet dataset
        molfeat_featurizer: MolFeat featurizer
        output_type: Type of output
        target_dict: Dictionary of target options when providing ChEMBL names for lookup in other datasets
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
            #, dtype = torch.float32 didn't work
        ),
        output_type: str = 'concat',
        target_dict: Optional[TypedDict] = None,
    ):
        self.molfeat_featurizer = molfeat_featurizer
        self.smiles_one = molfeat_featurizer(mmp_df[smiles_one].values)
        self.smiles_two = molfeat_featurizer(mmp_df[smiles_two].values)
        self.label = [int(label) for label in mmp_df[label].values]
        self.target = mmp_df[target].values
        self.output_type = output_type
        self.target_dict = target_dict

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        molecule_one = torch.as_tensor(self.smiles_one[idx], dtype = torch.float32)
        molecule_two = torch.as_tensor(self.smiles_two[idx], dtype = torch.float32)
        label = torch.as_tensor(self.label[idx], dtype = torch.long)
        #target = torch.from_numpy(self.target[idx]) protein target, add support later since "All" needs to be encoded
        if self.output_type == 'pair':
            return molecule_one, molecule_two, label#, target
        elif self.output_type == 'concat':
            return torch.cat((molecule_one, molecule_two), dim = -1), label #,target