from dataclasses import dataclass

@dataclass
class Paths:
    ## from default paths config
    root_dir: str
    data_dir: str
    log_dir: str
    output_dir: str
    work_dir: str

@dataclass
class Files:
    train_data: str
    train_labels: str
    test_data: str
    test_labels: str

@dataclass
class Data:
    filename: str
    data_dir: str
    batch_size: int
    train_val_test_split: List[int]
    num_workers: int
    pin_memory: bool
    shuffle: bool
    output_type: str
    molfeat_featurizer: MoleculeTransformer
    target_dict: dict

@dataclass
class ACAconfig:
    """Groups parameters in terms of the previously
    defined dataclasses"""
    data: Data
    hydra: str
    model: str
    paths: Paths
    files: Files
    trainer: str
    wandb: str
    seed: int = 42

cs = ConfigStore.instance().store(
    name="ACAconfig",
    node=ACAconfig)
