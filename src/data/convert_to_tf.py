import os
from typing import Tuple

import pandas as pd
import tensorflow as tf

from ..config import INPUT, INTERIM_DATA_DIR, LABELS
from .preprocess import load_preprocessed_data


def main() -> None:
    # Load preprocessed dataframes
    train_df, val_df, test_df = load_preprocessed_data()
    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

    # Create TensorFlow datasets from the dataframes
    train_ds = df_to_tf_dataset(train_df)
    val_ds = df_to_tf_dataset(val_df)
    test_ds = df_to_tf_dataset(test_df)

    # retrieve a batch (of 32 reviews and labels) from the dataset
    text_batch, label_batch = next(iter(train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print("Review", first_review)
    print("Label", first_label)

    # Save the datasets
    save_tf_datasets(train_ds, val_ds, test_ds)


def df_to_tf_dataset(
    df: pd.DataFrame,
    inputs: list = INPUT,
    labels: list = LABELS,
    batch_size: int = 32,
    shuffle: bool = True,
) -> tf.data.Dataset:
    features = df[inputs].values
    targets = df[labels].values
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    return dataset.batch(batch_size, drop_remainder=True)


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
    print(f"TF datasets saved to {output_dir}.")


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
