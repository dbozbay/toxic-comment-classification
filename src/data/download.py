# import os
from typing import Tuple

import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from kagglehub.exceptions import KaggleApiHTTPError

# from ..config import DATASET, RAW_DATA_PATH



def load_data(handle: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _load_train(handle), _load_test(handle), _load_test_labels(handle)

def _load_train(handle: str) -> pd.DataFrame:
    try:
        return kagglehub.load_dataset(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=handle,
            path="train.csv",
        )
    except KaggleApiHTTPError as e:
        raise ValueError(f"Error loading train data: {e}")


def _load_test(handle: str) -> pd.DataFrame:
    try:
        return kagglehub.load_dataset(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=handle,
            path="test.csv",
        )
    except KaggleApiHTTPError as e:
        raise ValueError(f"Error loading test data: {e}")


def _load_test_labels(handle: str) -> pd.DataFrame:
    try:
        return kagglehub.load_dataset(
            adapter=KaggleDatasetAdapter.PANDAS,
            handle=handle,
            path="test_labels.csv",
        )
    except KaggleApiHTTPError as e:
        raise ValueError(f"Error loading test labels data: {e}")


# def _raw_train_is_valid(train_df: pd.DataFrame) -> bool:
#     if not isinstance(train_df, pd.DataFrame):
#         raise TypeError("The raw train data is not a DataFrame.")
#     if len(train_df) == 0:
#         raise ValueError("The raw train data is empty.")
#     if "id" not in train_df.columns:
#         raise ValueError("The raw train data does not contain an 'id' column.")
#     if INPUT not in train_df.columns:
#         raise ValueError(f"The raw train data does not contain an '{INPUT}' column.")
#     if not all(label in train_df.columns for label in LABELS):
#         raise ValueError("The raw train data does not contain all the label columns.")

#     expected_columns = {"id", INPUT} | set(LABELS)
#     actual_columns = set(train_df.columns)
#     if actual_columns != expected_columns:
#         raise ValueError(
#             f"The raw train data contains unexpected columns. Expected columns: {expected_columns}, but got: {actual_columns}"
#         )

#     return True


# def _raw_test_is_valid(test_df: pd.DataFrame) -> bool:
#     if not isinstance(test_df, pd.DataFrame):
#         raise TypeError("The raw test data is not a DataFrame.")
#     if len(test_df) == 0:
#         raise ValueError("The raw test data is empty.")
#     if "id" not in test_df.columns:
#         raise ValueError("The raw test data does not contain an 'id' column.")
#     if INPUT not in test_df.columns:
#         raise ValueError(f"The raw test data does not contain an '{INPUT}' column.")

#     expected_columns = {"id", INPUT}
#     actual_columns = set(test_df.columns)
#     if actual_columns != expected_columns:
#         raise ValueError(
#             f"The raw test data contains unexpected columns. Expected columns: {expected_columns}, but got: {actual_columns}"
#         )

#     return True


# def _raw_test_labels_is_valid(test_labels_df: pd.DataFrame) -> bool:
#     if not isinstance(test_labels_df, pd.DataFrame):
#         raise TypeError("The raw test labels data is not a DataFrame.")
#     if len(test_labels_df) == 0:
#         raise ValueError("The raw test labels data is empty.")
#     if "id" not in test_labels_df.columns:
#         raise ValueError("The raw test labels data does not contain an 'id' column.")
#     if not all(label in test_labels_df.columns for label in LABELS):
#         raise ValueError(
#             "The raw test labels data does not contain all the label columns."
#         )

#     expected_columns = {"id"} | set(LABELS)
#     actual_columns = set(test_labels_df.columns)
#     if actual_columns != expected_columns:
#         raise ValueError(
#             f"The raw test labels data contains unexpected columns. Expected columns: {expected_columns}, but got: {actual_columns}"
#         )

#     return True


# def load_raw_data(
#     download_path: str = RAW_DATA_DIR,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """Loads the raw Kaggle datasets."""
#     if not _raw_data_exists(download_path):
#         raise FileNotFoundError(
#             f"Raw data files not found in '{download_path}'. Please download the data first using `download_data()`."
#         )
#     train_df = pd.read_csv(os.path.join(download_path, "train.csv"))
#     test_df = pd.read_csv(os.path.join(download_path, "test.csv"))
#     test_labels_df = pd.read_csv(os.path.join(download_path, "test_labels.csv"))

#     return train_df, test_df, test_labels_df


# if __name__ == "__main__":
#     main()
