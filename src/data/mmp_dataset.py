class MMPDataset(Dataset):
    def __init__(self,
        smiles: str,
        label: str,
        target: int,
        target_dict: dict,
        featurizer: str = 'morgan',
        fp_len: int = 2048,
        smiles_transform,
        sanitize: bool = True,
    ## add possibility of decoding target directly into
    ## the text included in the ACNet file

    ## use molfeat to obtain descriptors from sequences
    ):
        self.smiles = smiles
        self.target = target
        self.smiles_transform = smiles_transform
    
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        target = self.target[idx]
        smiles = self.smiles_transform(smiles)
        return smiles, target

