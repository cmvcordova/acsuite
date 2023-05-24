import json
import pandas as pd
import datamol as dm
from typing import Union, List

def read_ACNet_single_line_JSON_file(path_to_json_file: str) -> pd.DataFrame:
    """
    Read MMP dataset from JSON file as provided after generating ACNet datasets
    as generated from https://github.com/DrugAI/ACNet#usage after running
    GenerateACDatasets and convert it to a pandas DataFrame.
    
    GenerateACDatasets generates a single line JSON file, delimiting MMPs by a comma.
    
    Args:
        input_csv (str): Path to the JSON file.
    Returns:
        pd.DataFrame: MMP dataset.
    """
    
    df_list = []

    with open(path_to_json_file) as source:
        data = json.load(source)

    for key in data.keys():
        _df = pd.json_normalize(data, record_path=[key])
        _df['Target'] = key
        df_list.append(_df)
    df = pd.concat(df_list)
    return df

def sanitize_standardize(smiles_list: List[str]): -> List[rdkit.Chem.rdchem.Mol]
   """ 
   convert a list of SMILES strings to a list of sanitized and standardized 
   RDKit molecules with datamol.   
    Args:
        smiles_list: List of SMILES strings.
    Returns:
        standardized_mols: List of standardized RDKit molecules.
    """
  standardized_mols = []
  for i in range(len(smiles_list)):
    with dm.without_rdkit_log():
      _mol = dm.to_mol(smiles_list[i])
      _mol = dm.fix_mol(_mol)
      _mol = dm.sanitize_mol(_mol)
      _mol = dm.standardize_mol(_mol)
      standardized_mols.append(_mol)
  return standardized_mols 

def descriptors_from_smiles(smiles_array:)