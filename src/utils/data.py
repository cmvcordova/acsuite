import json
import pandas as pd

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