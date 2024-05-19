
# standard python/pytorch imports
import pandas as pd
import torch
import numpy as np
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
        
        print("Total MMPs: ", len(mmp_df))
        num_positive = len(mmp_df[mmp_df[label] == '1'])
        num_negative = len(mmp_df[mmp_df[label] == '0'])
        print(f"Number of AC pairs: {num_positive} ({num_positive / len(mmp_df) * 100:.2f}%)")
        print(f"Number of Non AC pairs: {num_negative} ({num_negative / len(mmp_df) * 100:.2f}%)")
        
        if filter_type == 'positive':
            print("Filtering for AC pairs...")
            mmp_df = mmp_df[mmp_df[label] == '1']
        elif filter_type == 'negative':
            print("Filtering for Non AC pairs...")
            mmp_df = mmp_df[mmp_df[label] == '0']
            
        featurized_smiles_one_arrays = [self.molfeat_featurizer(smile) for smile in mmp_df[smiles_one].values]
        featurized_smiles_two_arrays = [self.molfeat_featurizer(smile) for smile in mmp_df[smiles_two].values]

        if not featurized_smiles_one_arrays:
            raise ValueError(f"No data after filtering for {filter_type} pairs.")
        
        self.featurized_smiles_one = torch.tensor(np.stack(featurized_smiles_one_arrays), dtype=torch.float32)
        self.featurized_smiles_two = torch.tensor(np.stack(featurized_smiles_two_arrays), dtype=torch.float32)
        self.labels = torch.tensor(mmp_df[label].astype(int).values, dtype=torch.long)
        self.target = mmp_df[target].values
                
        if input_type == 'single':
            # Combine SMILES from pairs, then featurize each individually.
            self.featurized_smiles = torch.cat((self.featurized_smiles_one, self.featurized_smiles_two), dim=0)
            self.labels = torch.cat((self.labels, self.labels), dim=0)
            
        if input_type == 'concat':
            self.featurized_pairs = torch.cat((self.featurized_smiles_one, self.featurized_smiles_two), dim=-1)

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        if self.input_type == 'single':
            return self.featurized_smiles[idx], self.labels[idx]
        elif self.input_type == 'pair':
            return self.featurized_smiles_one[idx], self.featurized_smiles_two[idx], self.labels[idx]
        elif self.input_type == 'concat':
            return self.featurized_pairs[idx], self.labels[idx]

        #target = torch.from_numpy(self.target[idx]) protein target, add support later since "All" needs to be encoded
