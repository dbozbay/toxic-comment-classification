import os
import zipfile

from dotenv import load_dotenv

COMPETITION = "jigsaw-toxic-comment-classification-challenge"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
COMPETITION_DIR = os.path.join(BASE_DIR, "data", COMPETITION)
RAW_DATA_DIR = os.path.join(COMPETITION_DIR, "raw")


def download_competition_dataset(
    competition: str,
    download_path: str,
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

    print(
        f"Downloading datasets from the '{competition}' competition to '{download_path}'..."
    )

    api.competition_download_files(
        competition=competition, path=download_path, quiet=False
    )

    print(
        f"Datasets from '{competition}' downloaded successfully to '{download_path}'."
    )

    # Unzip downloaded files if `unzip` is True
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


if __name__ == "__main__":
    try:
        download_competition_dataset(COMPETITION, RAW_DATA_DIR)
    except Exception as e:
        print(f"An error occurred: {e}")
