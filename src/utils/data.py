import torch
import JSON
import pandas as pd

def read_ACNet_single_line_JSON_datset(path_to_file: str, targets: int) -> pd.DataFrame:
    """
    Read MMP dataset from JSON file as provided after generating ACNet datasets.
    as detailed in https://github.com/DrugAI/ACNet#usage
    Args:
        input_csv (str): Path to the JSON file.
    Returns:
        pd.DataFrame: MMP dataset.
    """
    df = print(path_to_file)
    #df = pd.read_csv(input_csv)
    return df