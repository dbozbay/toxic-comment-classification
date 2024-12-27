# tests/test_download.py
import os
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.download import download_data, load_raw_data


@pytest.fixture
def mock_kaggle_download():
    with patch("src.data.download.kagglehub.dataset_download") as mock_download:
        mock_download.return_value = "/mock/temp_path"
        yield mock_download

def test_download_data_success(mock_kaggle_download, tmp_path):
    # Setup
    dataset = "mock/dataset"
    download_path = tmp_path / "data"
    os.makedirs("/mock/temp_path", exist_ok=True)

    # Mock the presence of files in the temporary download path
    with patch("os.listdir", return_value=["train.csv", "test.csv", "test_labels.csv", "sample_submission.csv"]):
        with patch("os.path.isfile", return_value=True):
            with patch("shutil.move") as mock_move:
                with patch("shutil.rmtree") as mock_rmtree:
                    # Execute
                    download_data(dataset=dataset, download_path=str(download_path))

                    # Assertions
                    mock_kaggle_download.assert_called_once_with(dataset, force_download=True)
                    assert download_path.exists()
                    mock_move.assert_any_call(
                        os.path.join("/mock/temp_path", "train.csv"),
                        str(download_path / "train.csv"),
                    )
                    mock_move.assert_any_call(
                        os.path.join("/mock/temp_path", "test.csv"),
                        str(download_path / "test.csv"),
                    )
                    mock_move.assert_any_call(
                        os.path.join("/mock/temp_path", "test_labels.csv"),
                        str(download_path / "test_labels.csv"),
                    )
                    # sample_submission.csv should not be moved
                    assert not mock_move.called_with(
                        os.path.join("/mock/temp_path", "sample_submission.csv"), str(download_path / "sample_submission.csv")
                    )
                    mock_rmtree.assert_called_once_with("/mock/temp_path")

def test_download_data_type_errors():
    with pytest.raises(TypeError):
        download_data(dataset=123, download_path="valid/path")

    with pytest.raises(TypeError):
        download_data(dataset="valid/dataset", download_path=456)

def test_load_raw_data_success(mock_kaggle_download, tmp_path):
    # Setup
    download_path = tmp_path / "data"
    download_path.mkdir(parents=True, exist_ok=True)

    # Create mock CSV files
    train_csv = download_path / "train.csv"
    test_csv = download_path / "test.csv"
    test_labels_csv = download_path / "test_labels.csv"
    for file in [train_csv, test_csv, test_labels_csv]:
        file.touch()
        pd.DataFrame({"col": [1, 2, 3]}).to_csv(file, index=False)

    # Execute
    train_df, test_df, test_labels_df = load_raw_data(download_path=str(download_path))

    # Assertions
    assert not mock_kaggle_download.called
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(test_labels_df, pd.DataFrame)
    assert not train_df.empty
    assert not test_df.empty
    assert not test_labels_df.empty
