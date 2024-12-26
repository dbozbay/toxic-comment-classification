import os
import shutil
from typing import Tuple

import kagglehub
import pandas as pd

from src.config import COMPETITION, RAW_DATA_DIR


def download_data(
    competition: str = COMPETITION, download_path: str = RAW_DATA_DIR
) -> None:
    if not isinstance(competition, str):
        raise TypeError(
            f"Expected 'competition' to be a str, got {type(competition).__name__}"
        )
    if not isinstance(download_path, str):
        raise TypeError(
            f"Expected 'download_path' to be a str, got {type(download_path).__name__}"
        )

    temp_path = kagglehub.dataset_download(competition, force_download=True)

    print("Path to dataset files:", temp_path)

    # Move files to the desired download path
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for file_name in os.listdir(temp_path):
        full_file_name = os.path.join(temp_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.move(full_file_name, download_path)
    shutil.rmtree(temp_path)

    print("Files moved to:", RAW_DATA_DIR)


def load_raw_data(
    download_path: str = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(download_path, str):
        raise TypeError(
            f"Expected 'download_path' to be a str, got {type(download_path).__name__}"
        )

    raw_data_files = ["train.csv", "test.csv", "test_labels.csv"]

    if not all(
        os.path.exists(os.path.join(download_path, file)) for file in raw_data_files
    ):
        print("Raw data files not found. Initiating download...")
        download_data(download_path=download_path)

    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(download_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(download_path, "test.csv"))
    test_labels_df = pd.read_csv(os.path.join(download_path, "test_labels.csv"))

    return train_df, test_df, test_labels_df


if __name__ == "__main__":
    train, test, test_labels = load_raw_data()
    print(train.head())
