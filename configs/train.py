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
class ACAconfig:
    paths: Paths
    files: Files
