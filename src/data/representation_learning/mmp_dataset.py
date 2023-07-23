
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
        positive_only: Whether to only include positive samples i.e. activity cliffs
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
        positive_only: bool = False,
        molfeat_featurizer = MoleculeTransformer(
            FPCalculator('ecfp', 
            length = 2048,
            radius = 4)
            #, dtype = torch.float32 didn't work
        ),
        input_type: str = 'concat',
        target_dict: Optional[TypedDict] = None
    ):
        self.molfeat_featurizer = molfeat_featurizer

        if positive_only:
            mmp_df = mmp_df[mmp_df[label] == 1]
            
        self.featurized_smiles_one = molfeat_featurizer(mmp_df[smiles_one].values)
        self.featurized_smiles_two = molfeat_featurizer(mmp_df[smiles_two].values)
        self.label = [int(label) for label in mmp_df[label].values]
        self.target = mmp_df[target].values
        self.input_type = input_type
        self.target_dict = target_dict

        self.featurized_smiles = None
        if input_type == 'single':
            self.featurized_smiles = self.featurized_smiles_one + self.featurized_smiles_two

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.featurized_smiles:
            molecule = torch.as_tensor(self.featurized_smiles[idx], dtype = torch.float32)

        else:
            molecule_one = torch.as_tensor(self.featurized_smiles_one[idx], dtype = torch.float32)
            molecule_two = torch.as_tensor(self.featurized_smiles_two[idx], dtype = torch.float32)
            label = torch.tensor(self.label[idx], dtype = torch.long)
        #target = torch.from_numpy(self.target[idx]) protein target, add support later since "All" needs to be encoded
        if self.input_type == 'pair':
            return molecule_one, molecule_two, label#, target
        elif self.input_type == 'concat':
            return torch.cat((molecule_one, molecule_two), dim = -1), label #,target
        elif self.input_type == 'single':
            ## label not supported since it denotes AC relationship between MMPs
            return molecule #, target