
# standard python/pytorch imports
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional

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
        target_dict: Optional[dict] = None,
        featurizer: str = 'ecfp',
        fp_len: int = 2048,
        sanitize: bool = True
    ):
        self.smiles_one = mmp_df[smiles_one]
        self.smiles_two = mmp_df[smiles_two]
        self.label = mmp_df[label]
        self.target = mmp_df[target]
        self.target_dict = target_dict
        self.featurizer = featurizer
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        smile_one = self.smiles_one[idx]
        smile_two = self.smiles_two[idx]
        target = self.target[idx]
        label = self.label[idx]
        return smile_one, smile_two, target, label