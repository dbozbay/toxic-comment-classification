import os
import shutil
from typing import Tuple

import kagglehub
import pandas as pd

from ..config import DATASET_HANDLER, INPUT, LABELS, RAW_DATA_DIR


def main() -> None:
    if not raw_data_exists():
        print("Raw data files not found. Initiating download...")
        download_data()
    else:
        print("Raw data already exists.")


def download_data(
    dataset: str = DATASET_HANDLER, download_path: str = RAW_DATA_DIR
) -> None:
    temp_path = kagglehub.dataset_download(dataset, force_download=True)

    print(f"Files downloaded to {temp_path}. Moving to {download_path}...")
    _move_files(temp_path, download_path)
    shutil.rmtree(temp_path)

    print("Validating raw data...")
    train_df, test_df, test_labels_df = load_raw_data(download_path)
    _validate_raw_data(train_df, test_df, test_labels_df)

    print("Download complete.")


def load_raw_data(
    download_path: str = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the raw Kaggle datasets."""
    if not raw_data_exists(download_path):
        raise FileNotFoundError(
            f"Raw data files not found in '{download_path}'. Please download the data first using `download_data()`."
        )
    train_df = pd.read_csv(os.path.join(download_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(download_path, "test.csv"))
    test_labels_df = pd.read_csv(os.path.join(download_path, "test_labels.csv"))

    return train_df, test_df, test_labels_df


def raw_data_exists(download_path: str = RAW_DATA_DIR) -> bool:
    raw_data_files = ["train.csv", "test.csv", "test_labels.csv"]
    return all(
        os.path.exists(os.path.join(download_path, file)) for file in raw_data_files
    )


def _move_files(temp_path: str, download_path: str) -> None:
    os.makedirs(download_path, exist_ok=True)
    for file_name in os.listdir(temp_path):
        full_file_name = os.path.join(temp_path, file_name)
        if os.path.isfile(full_file_name) and file_name != "sample_submission.csv":
            shutil.move(full_file_name, download_path)


def _validate_raw_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, test_labels_df: pd.DataFrame
) -> None:
    _validate_dataframe(train_df, "train")
    _validate_dataframe(test_df, "test")
    _validate_dataframe(test_labels_df, "test_labels")


def _validate_dataframe(df: pd.DataFrame, df_type: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"The '{df_type}' parameter must be a DataFrame.")
    if len(df) == 0:
        raise ValueError(f"The {df_type} DataFrame must not be empty.")
    if "id" not in df.columns:
        raise ValueError(f"The {df_type} DataFrame must contain an 'id' column.")
    if df_type != "test_labels" and INPUT not in df.columns:
        raise ValueError(f"The {df_type} DataFrame must contain an '{INPUT}' column.")
    if df_type != "test" and not all(label in df.columns for label in LABELS):
        raise ValueError(f"The {df_type} DataFrame must contain all the label columns.")


if __name__ == "__main__":
    main()
