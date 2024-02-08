from typing import Any, Dict, Optional, Tuple, Union, List

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.data.property_prediction.moleculeace_dataset import MoleculeACEDataset

class MoleculeACEDataModule(LightningDataModule):
    """
    Lightning DataModule for MoleculeACE datasets
    
    Args:

        dataset_name: Name of the dataset as included in MoleculeACE
        task: Task to perform, e.g. 'classification' or 'regression', is supplied to dataset
        molfeat_featurizer: MolFeat featurizer

        batch_size (int): Batch size
        num_workers (int): Number of workers
        pin_memory (bool): Whether to pin memory
        shuffle (bool): Whether to shuffle the data
    """
    
    def __init__(
        self, 
        ## dataset options
        dataset_name: str, ## todo: extend to using lightning's CombinedLoader for multiple datasets?
        task: str,
        molfeat_featurizer,
        # misc parameters
        batch_size: int, 
        num_workers: int,
        pin_memory: bool, 
        shuffle: bool,
        train_val_split: Optional[float] = 0.8

    ):
        super().__init__()
        ## this line allows to access init params with 'self.hparams' attribute
        ## also ensures init params will be stored in ckpt

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

    def prepare_data(self) -> None:
        print('Preprocessing data...')
        """ Download data if needed

        Do not use it to assign state (self.x = y). """

        ## add additional split info here
        
    def setup(self, seed: int = 42, stage=None):
        print('Setting up data...')
        if not self.data_train and not self.data_val and not self.data_test:
            print(self.hparams.dataset_name)

            full_train_dataset = MoleculeACEDataset(self.hparams.dataset_name, 
            data_split='train', 
            task=self.hparams.task, 
            molfeat_featurizer=self.hparams.molfeat_featurizer)

            test_dataset = MoleculeACEDataset(self.hparams.dataset_name, 
            data_split='test', 
            task=self.hparams.task,
            molfeat_featurizer=self.hparams.molfeat_featurizer)

            # Splitting full_train_dataset into train and validation
            train_size = int(self.hparams.train_val_split * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size

            self.data_train, self.data_val = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
            self.data_test = test_dataset
                
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory
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
    _ = MoleculeACEDataModule()
