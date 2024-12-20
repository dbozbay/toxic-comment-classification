import os
from typing import Tuple

import pandas as pd
import tensorflow as tf

from ..config import INPUT, INTERIM_DATA_DIR, LABELS
from .preprocess import load_preprocessed_data


def main():
    train_df, val_df, test_df = load_preprocessed_data()

    train_ds = df_to_tf_dataset(train_df)
    val_ds = df_to_tf_dataset(val_df)
    test_ds = df_to_tf_dataset(test_df)

    print("Training: ", len(train_ds))
    print("Validation: ", len(val_ds))
    print("Test: ", len(test_ds))

    save_tf_datasets(train_ds, val_ds, test_ds)


def df_to_tf_dataset(
    df: pd.DataFrame,
    inputs: list = INPUT,
    labels: list = LABELS,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Converts a DataFrame into a TensorFlow dataset thats optimized for learning."""
    features = df[inputs].values
    targets = df[labels].values
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def save_tf_datasets(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    output_dir: str = INTERIM_DATA_DIR,
) -> None:
    """Saves the preprocessed TensorFlow datasets."""
    os.makedirs(output_dir, exist_ok=True)
    train_ds.save(os.path.join(output_dir, "train_ds"), compression="GZIP")
    val_ds.save(os.path.join(output_dir, "val_ds"), compression="GZIP")
    test_ds.save(os.path.join(output_dir, "test_ds"), compression="GZIP")


def load_tf_datasets(
    input_dir: str = INTERIM_DATA_DIR,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_ds = tf.data.Dataset.load(
        os.path.join(input_dir, "train_ds"), compression="GZIP"
    )
    val_ds = tf.data.Dataset.load(os.path.join(input_dir, "val_ds"), compression="GZIP")
    test_ds = tf.data.Dataset.load(
        os.path.join(input_dir, "test_ds"), compression="GZIP"
    )
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    main()
