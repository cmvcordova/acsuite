
# standard python/pytorch imports
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Literal, Optional

## Molfeat related imports
import datamol as dm
from molfeat.calc import FPCalculator
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
        input_type: Specifies the way pairs in the dataset will be input to the model.
        filter_type: Specifies whether to include only positive or negative pairs.
    """
    def __init__(self,
        mmp_df: pd.DataFrame,
        smiles_one: str,
        smiles_two: str,
        label: str,
        target: str,
        molfeat_featurizer = MoleculeTransformer(
            FPCalculator('ecfp', 
            length = 2048,
            radius = 4)
            #, dtype = torch.float32 didn't work
        ),
        input_type: Literal['single', 'pair', 'concat'] = None,
        filter_type: Optional[Literal['positive', 'negative']] = None,
    ):
        ## input checks
        if input_type not in ['single', 'pair', 'concat']:
            raise ValueError(f"Unsupported input_type: {input_type}. Choose from 'single', 'pair', 'concat'.")

        if filter_type not in [None, 'positive', 'negative']:
            raise ValueError(f"Unsupported filter_type: {filter_type}. Choose from None, 'positive', 'negative'.")

        self.molfeat_featurizer = molfeat_featurizer
        self.input_type = input_type

        if filter_type == 'positive':
            mmp_df = mmp_df[mmp_df[label] == 1]
        elif filter_type == 'negative':
            mmp_df = mmp_df[mmp_df[label] == 0]
         
        self.featurized_smiles_one = molfeat_featurizer(mmp_df[smiles_one].values)
        self.featurized_smiles_two = molfeat_featurizer(mmp_df[smiles_two].values)
        
        self.labels = torch.tensor([int(lbl) for lbl in mmp_df[label].values], dtype=torch.long)
        self.target = mmp_df[target].values
 
        if input_type == 'single':
        # Combine SMILES from pairs, then featurize each individually.
            self.featurized_smiles = torch.cat((self.featurized_smiles_one, self.featurized_smiles_two), dim=0)
            self.labels = torch.cat((self.labels, self.labels), dim=0)

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        if self.input_type == 'single':
            return self.featurized_smiles[idx], self.labels[idx]
        else:
            molecule_one = self.featurized_smiles_one[idx]
            molecule_two = self.featurized_smiles_two[idx]
            label = self.labels[idx]
            if self.input_type == 'pair':
                return molecule_one, molecule_two, label
            elif self.input_type == 'concat':
                return torch.cat((molecule_one, molecule_two), dim = -1), label
        #target = torch.from_numpy(self.target[idx]) protein target, add support later since "All" needs to be encoded
