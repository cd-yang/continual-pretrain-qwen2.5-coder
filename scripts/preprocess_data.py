import os

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(file_path):
    """
    Load raw data from a CSV file.

    Args:
        file_path (str): Path to the raw data file.

    Returns:
        pd.DataFrame: Loaded raw data.
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Preprocess the raw data.

    Args:
        df (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Example preprocessing steps
    df = df.dropna()  # Remove missing values
    df = df[df["text"].str.len() > 0]  # Remove empty text entries
    return df


def save_preprocessed_data(df, file_path):
    """
    Save preprocessed data to a CSV file.

    Args:
        df (pd.DataFrame): Preprocessed data.
        file_path (str): Path to save the preprocessed data.
    """
    df.to_csv(file_path, index=False)


def main():
    raw_data_path = os.path.join("data", "raw_data.csv")
    preprocessed_data_path = os.path.join("data", "preprocessed_data.csv")

    # Load raw data
    raw_data = load_raw_data(raw_data_path)

    # Preprocess data
    preprocessed_data = preprocess_data(raw_data)

    # Save preprocessed data
    save_preprocessed_data(preprocessed_data, preprocessed_data_path)


if __name__ == "__main__":
    main()
