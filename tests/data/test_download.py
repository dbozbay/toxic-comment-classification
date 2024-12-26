import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.download import COMPETITION, RAW_DATA_DIR, download_data, load_raw_data


def test_download_data_type_error():
    with pytest.raises(TypeError):
        download_data(competition=123)  # Invalid type for competition
    with pytest.raises(TypeError):
        download_data(download_path=456)  # Invalid type for download_path


def test_load_raw_data_type_error():
    with pytest.raises(TypeError):
        load_raw_data(download_path=789)  # Invalid type for download_path


@patch("src.data.download.kagglehub.dataset_download")
@patch("src.data.download.shutil.move")
@patch("src.data.download.shutil.rmtree")
def test_download_data(mock_rmtree, mock_move, mock_dataset_download):
    # Mock the dataset download to return a temporary path
    temp_path = "/tmp/kaggle_dataset"
    mock_dataset_download.return_value = temp_path

    # Create a temporary directory and files to simulate the downloaded dataset
    os.makedirs(temp_path, exist_ok=True)
    with open(os.path.join(temp_path, "train.csv"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(temp_path, "test.csv"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(temp_path, "test_labels.csv"), "w") as f:
        f.write("dummy data")

    # Call the function
    download_data(competition=COMPETITION, download_path=RAW_DATA_DIR)

    # Check that the files were moved
    mock_move.assert_any_call(os.path.join(temp_path, "train.csv"), RAW_DATA_DIR)
    mock_move.assert_any_call(os.path.join(temp_path, "test.csv"), RAW_DATA_DIR)
    mock_move.assert_any_call(os.path.join(temp_path, "test_labels.csv"), RAW_DATA_DIR)

    # Check that the temporary directory was removed
    mock_rmtree.assert_called_once_with(temp_path)


@patch("src.data.download.download_data")
@patch("pandas.read_csv")
def test_load_raw_data(mock_read_csv, mock_download_data):
    # Mock the read_csv function to return dummy dataframes
    mock_read_csv.side_effect = [
        pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
        pd.DataFrame({"col1": [5, 6], "col2": [7, 8]}),
        pd.DataFrame({"col1": [9, 10], "col2": [11, 12]}),
    ]

    # Create a temporary directory and files to simulate the raw data files
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    with open(os.path.join(RAW_DATA_DIR, "train.csv"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(RAW_DATA_DIR, "test.csv"), "w") as f:
        f.write("dummy data")
    with open(os.path.join(RAW_DATA_DIR, "test_labels.csv"), "w") as f:
        f.write("dummy data")

    # Call the function
    train_df, test_df, test_labels_df = load_raw_data(download_path=RAW_DATA_DIR)

    # Check that the dataframes were loaded correctly
    assert not train_df.empty
    assert not test_df.empty
    assert not test_labels_df.empty

    # Check that the read_csv function was called with the correct file paths
    mock_read_csv.assert_any_call(os.path.join(RAW_DATA_DIR, "train.csv"))
    mock_read_csv.assert_any_call(os.path.join(RAW_DATA_DIR, "test.csv"))
    mock_read_csv.assert_any_call(os.path.join(RAW_DATA_DIR, "test_labels.csv"))

    # Clean up the temporary files
    os.remove(os.path.join(RAW_DATA_DIR, "train.csv"))
    os.remove(os.path.join(RAW_DATA_DIR, "test.csv"))
    os.remove(os.path.join(RAW_DATA_DIR, "test_labels.csv"))


if __name__ == "__main__":
    pytest.main()
