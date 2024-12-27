import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from ..config import BATCH_SIZE, INPUT, LABELS, PROCESSED_DATA_DIR, RANDOM_STATE
from .download import load_raw_data


def main() -> None:
    if preprocessed_data_exists():
        print("Preprocessed data already exists.")
    else:
        print("Loading raw data...")
        raw_train_df, raw_test_df, raw_test_labels_df = load_raw_data()
        train_df, test_df = preprocess_dataframes(
            raw_train_df, raw_test_df, raw_test_labels_df
        )
        train_ds, test_ds = convert_to_tf(train_df), convert_to_tf(test_df)
        train_ds, val_ds = split_tf_dataset(train_ds)
        print("Preprocessing complete.")
        save_tf_datasets(train_ds, val_ds, test_ds)


def preprocess_dataframes(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    raw_test_labels_df: pd.DataFrame,
    input: str = INPUT,
    labels: List[str] = LABELS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("Preprocessing dataframes...")
    train_df = raw_train_df.set_index("id")
    test_df = raw_test_df.set_index("id").join(raw_test_labels_df.set_index("id"))
    test_df = test_df.loc[~(test_df[labels] == -1).all(axis=1)]
    _validate_preprocessed_data(train_df)
    _validate_preprocessed_data(test_df)
    return train_df, test_df


def split_tf_dataset(
    dataset: tf.data.Dataset, test_size: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Splits a TensorFlow dataset into training and testing (or validation) sets."""
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be a float value between 0 and 1.")
    print("Splitting TensorFlow dataset...")
    n_samples = len(dataset)
    n_test_samples = int(n_samples * test_size)
    dataset = dataset.shuffle(n_samples, seed=RANDOM_STATE)
    test_ds = dataset.take(n_test_samples)
    train_ds = dataset.skip(n_test_samples)
    return train_ds, test_ds


def save_tf_datasets(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    output_dir: str = PROCESSED_DATA_DIR,
) -> None:
    """Saves the preprocessed TensorFlow datasets."""
    os.makedirs(output_dir, exist_ok=True)
    train_ds.save(os.path.join(output_dir, "train_ds"), compression="GZIP")
    val_ds.save(os.path.join(output_dir, "val_ds"), compression="GZIP")
    test_ds.save(os.path.join(output_dir, "test_ds"), compression="GZIP")
    print(f"TF datasets saved to {output_dir}.")


def load_preprocessed_data(
    preprocessed_path: str = PROCESSED_DATA_DIR,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Loads the preprocessed TensorFlow datasets."""
    if not preprocessed_data_exists(preprocessed_path):
        raise FileNotFoundError(
            f"Preprocessed data not found in '{preprocessed_path}'."
        )
    print("Loading preprocessed datasets...")
    train_ds = tf.data.Dataset.load(
        os.path.join(preprocessed_path, "train_ds"), compression="GZIP"
    )
    val_ds = tf.data.Dataset.load(
        os.path.join(preprocessed_path, "val_ds"), compression="GZIP"
    )
    test_ds = tf.data.Dataset.load(
        os.path.join(preprocessed_path, "test_ds"), compression="GZIP"
    )
    return train_ds, val_ds, test_ds


def _validate_preprocessed_data(df: pd.DataFrame) -> None:
    """Performs basic validation on the preprocessed data."""
    print("Validating preprocessed data...")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected 'df' to be a DataFrame.")
    if len(df) == 0:
        raise ValueError("Expected 'df' to be non-empty.")
    if df.index.name != "id":
        raise ValueError("Expected 'id' to be the index.")
    if INPUT not in df.columns:
        raise ValueError(f"Expected the input '{INPUT}' to be a column.")
    if not all(label in df.columns for label in LABELS):
        raise ValueError("Expected all labels to be columns.")
    if not np.array(df[LABELS].isin([0, 1]).all(axis=0)).all():
        raise ValueError("Expected all labels to be binary (0, 1).")


def convert_to_tf(
    df: pd.DataFrame,
    input: str = INPUT,
    labels: List[str] = LABELS,
    batch_size: int = BATCH_SIZE,
) -> tf.data.Dataset:
    """Converts a DataFrame to a batched TensorFlow dataset."""
    print("Converting DataFrame to TensorFlow dataset...")
    features = df[input].values
    targets = df[labels].values
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def preprocessed_data_exists(
    preprocessed_path: str = PROCESSED_DATA_DIR,
) -> bool:
    """Checks if the preprocessed TensorFlow datasets exist."""
    if not tf.io.gfile.exists(preprocessed_path):
        return False
    preprocessed_dirs = ["train_ds", "val_ds", "test_ds"]
    return all(
        tf.io.gfile.exists(os.path.join(preprocessed_path, dir))
        for dir in preprocessed_dirs
    )


if __name__ == "__main__":
    main()
