import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.download import download_data, main

mock_train_data = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "comment_text": [
            "This is a comment",
            "This is another comment",
            "Yet another comment",
            "More comments here",
            "Final comment",
        ],
        "toxic": [0, 1, 0, 1, 0],
        "severe_toxic": [0, 0, 0, 0, 0],
        "obscene": [0, 1, 0, 1, 0],
        "threat": [0, 0, 0, 0, 0],
        "insult": [0, 1, 0, 1, 0],
        "identity_hate": [0, 0, 0, 0, 0],
    }
)

mock_test_data = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "comment_text": [
            "This is a test comment",
            "This is another test comment",
            "Yet another test comment",
            "More test comments here",
            "Final test comment",
        ],
    }
)

mock_test_labels_data = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "toxic": [-1, 0, -1, 0, -1],
        "severe_toxic": [-1, 0, -1, 0, -1],
        "obscene": [-1, 0, -1, 0, -1],
        "threat": [-1, 0, -1, 0, -1],
        "insult": [-1, 0, -1, 0, -1],
        "identity_hate": [-1, 0, -1, 0, -1],
    }
)

@pytest.fixture
def mock_kagglehub():
    with patch("kagglehub.load_dataset") as mock_load_dataset:
        mock_load_dataset.side_effect = [
            mock_train_data,
            mock_test_data,
            mock_test_labels_data,
        ]
        yield mock_load_dataset

@pytest.fixture
def mock_os_makedirs():
    with patch("os.makedirs") as mock_makedirs:
        yield mock_makedirs

@pytest.fixture
def mock_raw_data_exists():
    with patch("src.data.download._raw_data_exists") as mock_exists:
        yield mock_exists

@pytest.fixture
def mock_download_data():
    with patch("src.data.download.download_data") as mock_download:
        yield mock_download


def test_main_data_exists(mock_raw_data_exists, mock_download_data):
    mock_raw_data_exists.return_value = True
    main()
    mock_download_data.assert_not_called()

def test_main_data_not_exists(mock_raw_data_exists, mock_download_data):
    mock_raw_data_exists.return_value = False
    main()
    mock_download_data.assert_called_once()


def test_download_data_success(mock_kagglehub, mock_os_makedirs):
    with tempfile.TemporaryDirectory() as temp_dir:
        download_data(download_path=temp_dir)

        # Check that the directory was created
        mock_os_makedirs.assert_called_once_with(temp_dir, exist_ok=True)

        # Check that the files were created
        train_file = os.path.join(temp_dir, "train.csv")
        test_file = os.path.join(temp_dir, "test.csv")
        test_labels_file = os.path.join(temp_dir, "test_labels.csv")

        assert os.path.exists(train_file), f"{train_file} does not exist"
        assert os.path.exists(test_file), f"{test_file} does not exist"
        assert os.path.exists(test_labels_file), f"{test_labels_file} does not exist"

        # Check the contents of the files
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        test_labels_df = pd.read_csv(test_labels_file)

        pd.testing.assert_frame_equal(train_df, mock_train_data)
        pd.testing.assert_frame_equal(test_df, mock_test_data)
        pd.testing.assert_frame_equal(test_labels_df, mock_test_labels_data)


def test_download_data_invalid_train(mock_kagglehub):
    invalid_train_data = pd.DataFrame({"id": [1, 2]})  # Missing required columns
    mock_kagglehub.side_effect = [
        invalid_train_data,
        mock_test_data,
        mock_test_labels_data,
    ]
    with pytest.raises(
        ValueError,
        match="The raw train data does not contain an 'comment_text' column.",
    ):
        download_data()


def test_download_data_invalid_test(mock_kagglehub):
    invalid_test_data = pd.DataFrame({"id": [1, 2]})  # Missing required columns
    mock_kagglehub.side_effect = [
        mock_train_data,
        invalid_test_data,
        mock_test_labels_data,
    ]
    with pytest.raises(
        ValueError, match="The raw test data does not contain an 'comment_text' column."
    ):
        download_data()


def test_download_data_invalid_test_labels(mock_kagglehub):
    invalid_test_labels_data = pd.DataFrame({"id": [1, 2]})  # Missing required columns
    mock_kagglehub.side_effect = [
        mock_train_data,
        mock_test_data,
        invalid_test_labels_data,
    ]
    with pytest.raises(
        ValueError,
        match="The raw test labels data does not contain all the label columns.",
    ):
        download_data()
