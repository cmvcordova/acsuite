def read_mmp_datset(input_csv: str, targets: int) -> pd.DataFrame:
    """
    Read MMP dataset from CSV file.
    Args:
        input_csv (str): Path to the CSV file.
    Returns:
        pd.DataFrame: MMP dataset.
    """
    df = pd.read_csv(input_csv)
    return df