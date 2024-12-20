import os
from typing import Tuple

import pandas as pd

from ..config import INPUT, LABELS, PROCESSED_DATA_DIR
from .download import load_raw_data


def main():
    raw_train, raw_test, raw_test_labels = load_raw_data()
    train_df, val_df, test_df = preprocess_data(
        raw_train, raw_test, raw_test_labels, inputs=INPUT, labels=LABELS
    )
    save_preprocessed_data(train_df, val_df, test_df)


def preprocess_data(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    raw_test_labels_df: pd.DataFrame,
    inputs: list,
    labels: list,
    test_size: float = 0.2,
    random_state: int = 0,
):
    # Set `id` as the indices
    train_df = set_id_as_index(raw_train_df)
    test_df = set_id_as_index(raw_test_df)
    test_labels_df = set_id_as_index(raw_test_labels_df)

    # Merge test data with labels
    test_df = merge_on_index(test_df, test_labels_df)

    # Remove bad samples
    test_df = remove_bad_test_samples(test_df, labels)

    # Split train into training and validation sets
    train_df, val_df = split_train_validation(train_df)

    return train_df, val_df, test_df


def set_id_as_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sets the `id` column as the index of the DataFrame."""
    if "id" not in df.columns:
        raise KeyError("The DataFrame doe not contain an 'id'column.")
    return df.set_index("id", drop=True)


def merge_on_index(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merges two DataFrames using their indexes."""
    return pd.merge(df1, df2, left_index=True, right_index=True)


def remove_bad_test_samples(test_df: pd.DataFrame, labels: list) -> pd.DataFrame:
    """Removes samples in the test set that were not used for scoring. These samples are indicated with -1 for all labels."""
    return test_df.loc[~(test_df[labels] == -1).all(axis=1)]


def split_train_validation(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the traininig DataFrame into train and validation (or test) sets."""
    val_df = df.sample(frac=val_size, random_state=random_state)
    train_df = df.drop(val_df.index)
    return train_df, val_df


def save_preprocessed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = PROCESSED_DATA_DIR,
) -> None:
    """Saves preprocessed DataFrames as compressed CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(
        os.path.join(output_dir, "train.csv.gz"), index=True, compression="gzip"
    )
    val_df.to_csv(
        os.path.join(output_dir, "val.csv.gz"), index=True, compression="gzip"
    )
    test_df.to_csv(
        os.path.join(output_dir, "test.csv.gz"), index=True, compression="gzip"
    )


def load_preprocessed_data(
    input_dir: str = PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads preprocessed DataFrames from compressed CSV files."""
    train_df = pd.read_csv(
        os.path.join(input_dir, "train.csv.gz"), index_col=0, compression="gzip"
    )
    val_df = pd.read_csv(
        os.path.join(input_dir, "val.csv.gz"), index_col=0, compression="gzip"
    )
    test_df = pd.read_csv(
        os.path.join(input_dir, "test.csv.gz"), index_col=0, compression="gzip"
    )
    return train_df, val_df, test_df


if __name__ == "__main__":
    main()
