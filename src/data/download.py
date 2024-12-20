import os
import zipfile
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

from src.config import COMPETITION, RAW_DATA_DIR


def download_competition_data(
    competition: str = COMPETITION,
    download_path: str = RAW_DATA_DIR,
    unzip: bool = True,
) -> None:
    # Load environment variables from .env
    load_dotenv()

    # FIX: For some reason Kaggle's `__init__.py` script is immediately executed when imported,
    # which attempts to read environment variables before the .env is loaded.
    # This import statement can be moved to the top of the script once this is fixed.
    from kaggle import KaggleApi

    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise ValueError("KAGGLE_USERNAME or KAGGLE_KEY is missing from the .env file.")

    # Ensure environment variables are available to KaggleApi
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    print("Kaggle API authenticated.")

    # Ensure the download directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    api.competition_download_files(
        competition=competition, path=download_path, quiet=False
    )

    print(
        f"Datasets from '{competition}' downloaded successfully to '{download_path}'."
    )

    if unzip:
        print("Unzipping downloaded files...")
        for file_name in os.listdir(download_path):
            if file_name.endswith(".zip"):
                file_path = os.path.join(download_path, file_name)
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(download_path)
                print(f"Extracted: {file_name}")
                # Optionally, remove the zip file after extraction
                os.remove(file_path)
        print("Unzipping completed.")


def _check_raw_data_exists(download_path: RAW_DATA_DIR) -> bool:
    """Checks if the raw datasets exist in zipped format."""
    files_to_check = ["train.csv.zip", "test.csv.zip", "test_labels.csv.zip"]
    return all(
        os.path.exists(os.path.join(download_path, file_name))
        for file_name in files_to_check
    )


def load_raw_data(
    download_path: str = RAW_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not _check_raw_data_exists(download_path):
        download_competition_data(download_path)

    raw_train_df = pd.read_csv(os.path.join(download_path, "train.csv.zip"))
    raw_test_df = pd.read_csv(os.path.join(download_path, "test.csv.zip"))
    raw_test_labels_df = pd.read_csv(os.path.join(download_path, "test_labels.csv.zip"))

    return raw_train_df, raw_test_df, raw_test_labels_df


if __name__ == "__main__":
    try:
        download_competition_data()
    except Exception as e:
        print(f"An error occurred: {e}")
