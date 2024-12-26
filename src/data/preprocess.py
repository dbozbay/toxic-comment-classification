import os
from typing import Tuple

import pandas as pd

from ..config import INPUT, LABELS, PROCESSED_DATA_DIR
from .download import load_raw_data


def main() -> None:
    raw_train_df, raw_test_df, raw_test_labels_df = load_raw_data()
    train_df, val_df, test_df = preprocess_data(
        raw_train_df, raw_test_df, raw_test_labels_df, inputs=INPUT, labels=LABELS
    )
    _save_preprocessed_data(train_df, val_df, test_df)


def preprocess_data(
    raw_train_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    raw_test_labels_df: pd.DataFrame,
    inputs: list,
    labels: list,
    test_size: float = 0.2,
    random_state: int = 0,
):
    # Ensure inputs and labels are lists
    assert isinstance(inputs, list), "The 'inputs' parameter must be a list."
    assert isinstance(labels, list), "The 'labels' parameter must be a list."

    # Set `id` as the indices
    train_df = _set_id_as_index(raw_train_df)
    test_df = _set_id_as_index(raw_test_df)
    test_labels_df = _set_id_as_index(raw_test_labels_df)

    # Merge test data with labels
    test_df = _merge_on_index(test_df, test_labels_df)

    # Remove bad samples
    test_df = _remove_bad_test_samples(test_df, labels)

    # Split train into training and validation sets
    train_df, val_df = _split_train_validation(
        train_df, val_size=test_size, random_state=random_state
    )

    return train_df, val_df, test_df


def _set_id_as_index(df: pd.DataFrame) -> pd.DataFrame:
    assert "id" in df.columns, "The DataFrame must contain an 'id' column."
    return df.set_index("id", drop=True)


def _merge_on_index(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True)
    assert (
        len(merged_df) == len(df1) == len(df2)
    ), "Some rows were lost during the merge."
    return pd.merge(df1, df2, left_index=True, right_index=True)


def _remove_bad_test_samples(test_df: pd.DataFrame, labels: list) -> pd.DataFrame:
    return test_df.loc[~(test_df[labels] == -1).all(axis=1)]


def _split_train_validation(
    df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_df = df.sample(frac=val_size, random_state=random_state)
    train_df = df.drop(index=list(val_df.index))
    assert len(train_df) + len(val_df) == len(df)
    return train_df, val_df


def _save_preprocessed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = PROCESSED_DATA_DIR,
) -> None:
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
    print(f"Processed data saved to {PROCESSED_DATA_DIR}.")


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
