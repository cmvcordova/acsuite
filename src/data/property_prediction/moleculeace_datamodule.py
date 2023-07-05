from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.data.property_prediction.moleculeace_dataset import MoleculeACEDataset

class MoleculeACEDataModule(LightningDataModule):
    """
    Lightning DataModule for MMP AC datasets.
    
    Args:
        file_name: Name of the file
        data_dir (str): Path to the data
        train_val_test_split (Tuple[int, int, int]): Train, validation, and test split sizes in number of samples
        batch_size (int): Batch size
        num_workers (int): Number of workers
        pin_memory (bool): Whether to pin memory
        shuffle (bool): Whether to shuffle the data
        ## dataset options
        molfeat_featurizer: MolFeat featurizer
        output_type (str): Type of output
        ## Currently unused, to be added later
        target_dict (Dict[str, Any]): Dictionary of target options when providing ChEMBL names for lookup in other datasets
    """
    
    def __init__(
        self, 
        file_name: str,
        data_dir: str,
        train_val_test_split: Tuple[int, int, int],
        batch_size: int, 
        num_workers: int,
        pin_memory: bool, 
        shuffle: bool,
        ## dataset options
        molfeat_featurizer,
        input_type: str,
        target_dict: Dict[str, Any]
    ):
        super().__init__()
        ## this line allows to access init params with 'self.hparams' attribute
        ## also ensures init params will be stored in ckpt

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None



    def prepare_data(self) -> None:
        print('Preprocessing data...')
        """ Download data if needed

        Do not use it to assign state (self.x = y). """

        ## add additional split info here
        ## make a class similar to MNIST
        ## and add ACNet download method?
        
    def setup(self, seed: int = 42, stage=None):
        print('Setting up data...')
        if not self.data_train and not self.data_val and not self.data_test:
            self.hparams.train_val_test_split = [204,26,26] #mmp_dataset debug purposes
            mmp_df = read_ACNet_single_line_JSON_file(self.hparams.data_dir + self.hparams.file_name).iloc[:256] #debug purposes
            dataset = MMPDataset(mmp_df, 'SMILES1', 'SMILES2', 'Value', 'Target',
            input_type = self.hparams.input_type, 
            molfeat_featurizer = self.hparams.molfeat_featurizer,
            target_dict = self.hparams.target_dict)
            
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

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
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.shuffle
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
