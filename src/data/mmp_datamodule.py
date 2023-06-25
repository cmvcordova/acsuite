from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.utils.data import read_ACNet_single_line_JSON_file
from src.data.mmp_dataset import MMPDataset

class MMPDataModule(LightningDataModule):
    """
    PyTorch Lightning data module for MMP AC datasets.
    Args:
        data_path (str): Path to the data
        sequence_transform (callable): Transformation for the sequence
        cell_type_transform (callable): Transformation for the cell type
        batch_size (int): Batch size
        num_workers (int): Number of workers
    """
    
    def __init__(
        ## add type hints here
        self, 
        file_name,
        data_dir,
        train_val_test_split,
        batch_size, 
        num_workers,
        pin_memory, 
        shuffle,
        ## dataset options
        molfeat_featurizer,
        output_type,
        target_dict
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.file_name = file_name
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        ## dataset options
        self.molfeat_featurizer = molfeat_featurizer
        self.output_type = output_type
        self.target_dict = target_dict

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None



    def prepare_data(self) -> None:
        print('Preprocessing data...')
        ## add additional split info here
        ## make a class similar to MNIST
        ## and add ACNet download method?
        
    def setup(self, seed: int = 42, stage=None):
        print('Setting up data...')
        if not self.data_train and not self.data_val and not self.data_test:
            self.mmp_df = read_ACNet_single_line_JSON_file(self.data_dir + self.file_name)
            print(len(self.mmp_df))
            dataset = MMPDataset(self.mmp_df, 'SMILES1', 'SMILES2', 'Value', 'Target',
            output_type = self.output_type, 
            molfeat_featurizer = self.molfeat_featurizer,
            target_dict = self.target_dict)
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle
        )
    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
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
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading from checkpoint."""
        pass

if __name__ == '__main__':
    _ = MMPDataModule()



