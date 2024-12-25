import os
import zipfile
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

# FIX: Environment variables must be loaded before importing Kaggle API.
load_dotenv()

from kaggle import KaggleApi

from src.config import COMPETITION, RAW_DATA_DIR


def setup_kaggle_api() -> KaggleApi:
    """
    Sets up and authenticates the Kaggle API client using environment variables.
    Raises an error if authentication fails.
    """
    print("Authenticating Kaggle API...")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        raise ValueError(
            "Failed to authenticate Kaggle API. Ensure KAGGLE_USERNAME and KAGGLE_KEY are set."
        ) from e
    print("Kaggle API authenticated.")
    return api


def download_data(
    competition: str = COMPETITION,
    download_path: str = RAW_DATA_DIR,
    unzip: bool = True,
) -> None:
    """
    Downloads competition data from Kaggle and optionally unzips the files.
    """
    api = setup_kaggle_api()
    os.makedirs(download_path, exist_ok=True)  # Ensure directory exists
    api.competition_download_files(competition, path=download_path, quiet=False)

    if unzip:
        print("Unzipping downloaded files...")
        _unzip_files(download_path)
        print("Unzipping completed.")


def _unzip_files(directory: str) -> None:
    """
    Extracts all .zip files in a given directory and removes the zip files after extraction.
    """
    for file_name in os.listdir(directory):
        if file_name.endswith(".zip"):
            file_path = os.path.join(directory, file_name)
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(directory)
            os.remove(file_path)  # Clean up the zip file
            print(f"Extracted and removed: {file_name}")


def load_raw_data(
    download_path: str = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw train, test, and test label datasets as Pandas DataFrames.
    Downloads the datasets if not found in the specified path.
    """
    required_files = ["train.csv.zip", "test.csv.zip", "test_labels.csv.zip"]
    if not all(
        os.path.exists(os.path.join(download_path, file)) for file in required_files
    ):
        print("Required files not found. Initiating download...")
        download_data(download_path=download_path)

    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(download_path, "train.csv.zip"))
    test_df = pd.read_csv(os.path.join(download_path, "test.csv.zip"))
    test_labels_df = pd.read_csv(os.path.join(download_path, "test_labels.csv.zip"))
    print("Datasets loaded successfully.")
    return train_df, test_df, test_labels_df


if __name__ == "__main__":
    raw_train_df, raw_test_df, raw_test_labels_df = load_raw_data()
