import pandas as pd

def load_data(filepath):
    """
    Load the dataset from the given file path.
    Args:
        filepath (str): The path to the dataset.
    Returns:
        pd.DataFrame: The loaded dataset.
    """
    
    # filepath = 'r"../../data/raw/diabetes_dataset.csv"'
    df = pd.read_csv(filepath)
    # df.info()
    # df.describe()
    # print(df)
    return df

# df = pd.read_csv(r"C:\Users\luisz\Desktop\GASTRIC-CANCER2\diabetes_dataset.csv")
