import os
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import (
    INPUT,
    LABELS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)
from .download import _raw_data_exists


def main() -> None:
    raw_train_df, raw_test_df, raw_test_labels_df = load_raw_data()
    train_df, test_df = preprocess_data(raw_train_df, raw_test_df, raw_test_labels_df)

    processed_data_path = PROCESSED_DATA_PATH
    if _preprocessed_data_is_valid(train_df, test_df):
        os.makedirs(processed_data_path, exist_ok=True)
        train_df.to_csv(os.path.join(processed_data_path, "train.csv"))
        test_df.to_csv(os.path.join(processed_data_path, "test.csv"))
        print("Preprocessed data saved.")

    print("Raw training samples: ", len(raw_train_df))
    print("Raw testing samples: ", len(raw_test_df))
    print("Preprocessed training samples: ", len(train_df))
    print("Preprocessed testing samples: ", len(test_df))


def load_raw_data(
    raw_data_path: str = RAW_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the raw data from CSV files."""
    if not _raw_data_exists(raw_data_path):
        raise FileNotFoundError(
            f"Raw data not found in '{raw_data_path}'. Please download it first."
        )
    print("Loading raw data...")
    raw_train_df = pd.read_csv(os.path.join(raw_data_path, "train.csv"))
    raw_test_df = pd.read_csv(os.path.join(raw_data_path, "test.csv"))
    raw_test_labels_df = pd.read_csv(os.path.join(raw_data_path, "test_labels.csv"))
    return raw_train_df, raw_test_df, raw_test_labels_df


def preprocess_data(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    raw_test_labels_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Preprocessing dataframes...")
    train_df = raw_train_df.set_index("id")
    test_df = raw_test_df.set_index("id").join(raw_test_labels_df.set_index("id"))
    test_df = test_df.loc[~(test_df[LABELS] == -1).all(axis=1)]
    return train_df, test_df


def _preprocessed_data_is_valid(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Performs basic validation on the preprocessed data."""
    print("Validating preprocessed data...")
    if not isinstance(train_df, pd.DataFrame):
        raise TypeError("Expected 'train_df' to be a DataFrame.")
    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("Expected 'test_df' to be a DataFrame.")
    if len(train_df) == 0:
        raise ValueError("Expected 'train_df' to be non-empty.")
    if len(test_df) == 0:
        raise ValueError("Expected 'test_df' to be non-empty.")
    if train_df.index.name != "id":
        raise ValueError("Expected 'id' to be the index.")
    if test_df.index.name != "id":
        raise ValueError("Expected 'id' to be the index.")
    if INPUT not in train_df.columns:
        raise ValueError(f"Expected the input '{INPUT}' to be a column.")
    if not all(label in train_df.columns for label in LABELS):
        raise ValueError("Expected all labels to be columns.")
    if not np.array(train_df[LABELS].isin([0, 1]).all(axis=0)).all():
        raise ValueError("Expected all labels to be binary (0, 1).")
    if not np.array(test_df[LABELS].isin([0, 1]).all(axis=0)).all():
        raise ValueError("Expected all labels to be binary (0, 1).")

    expected_columns = {INPUT} | set(LABELS)
    actual_columns = set(train_df.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            f"The raw train data contains unexpected columns. Expected columns: {expected_columns}, but got: {actual_columns}"
        )

    print("Preprocessed data is valid.")
    return True


# def split_tf_dataset(
#     dataset: tf.data.Dataset, test_size: float = 0.2
# ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
#     """Splits a TensorFlow dataset into training and testing (or validation) sets."""
#     if test_size < 0 or test_size > 1:
#         raise ValueError("Test size must be a float value between 0 and 1.")
#     print("Splitting TensorFlow dataset...")
#     n_samples = len(dataset)
#     n_test_samples = int(n_samples * test_size)
#     dataset = dataset.shuffle(n_samples, seed=RANDOM_STATE)
#     test_ds = dataset.take(n_test_samples)
#     train_ds = dataset.skip(n_test_samples)
#     return train_ds, test_ds


# def _validate_preprocessed_data(df: pd.DataFrame) -> None:
#     """Performs basic validation on the preprocessed data."""
#     print("Validating preprocessed data...")
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Expected 'df' to be a DataFrame.")
#     if len(df) == 0:
#         raise ValueError("Expected 'df' to be non-empty.")
#     if df.index.name != "id":
#         raise ValueError("Expected 'id' to be the index.")
#     if INPUT not in df.columns:
#         raise ValueError(f"Expected the input '{INPUT}' to be a column.")
#     if not all(label in df.columns for label in LABELS):
#         raise ValueError("Expected all labels to be columns.")
#     if not np.array(df[LABELS].isin([0, 1]).all(axis=0)).all():
#         raise ValueError("Expected all labels to be binary (0, 1).")


# def convert_to_tf(
#     df: pd.DataFrame,
#     input: str = INPUT,
#     labels: List[str] = LABELS,
#     batch_size: int = BATCH_SIZE,
# ) -> tf.data.Dataset:
#     """Converts a DataFrame to a batched TensorFlow dataset."""
#     print("Converting DataFrame to TensorFlow dataset...")
#     features = df[input].values
#     targets = df[labels].values
#     dataset = tf.data.Dataset.from_tensor_slices((features, targets))
#     dataset = dataset.batch(batch_size, drop_remainder=True)
#     return dataset


# def preprocessed_data_exists(
#     preprocessed_path: str = PROCESSED_DATA_DIR,
# ) -> bool:
#     """Checks if the preprocessed TensorFlow datasets exist."""
#     if not tf.io.gfile.exists(preprocessed_path):
#         return False
#     preprocessed_dirs = ["train_ds", "val_ds", "test_ds"]
#     return all(
#         tf.io.gfile.exists(os.path.join(preprocessed_path, dir))
#         for dir in preprocessed_dirs
#     )


if __name__ == "__main__":
    main()
