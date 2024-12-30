# import os
# import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from kagglehub.exceptions import KaggleApiHTTPError

from src.data.download import _load_test, _load_test_labels, _load_train, load_data

from .mock_data import make_mock_data


@patch("src.data.download.kagglehub.load_dataset")
def test_load_data_valid_handle(mock_load_dataset):
    # Mock the return values for the valid handle
    mock_train_df, mock_test_df, mock_test_labels_df = make_mock_data()
    mock_load_dataset.side_effect = (mock_train_df, mock_test_df, mock_test_labels_df)

    result_train_df, result_test_df, result_test_labels_df = load_data(
        handle="valid_handle"
    )

    # Assertions to check if the dataframes are loaded correctly
    assert isinstance(
        result_train_df, pd.DataFrame
    ), "Expected train dataframe to be a pandas DataFrame"
    assert isinstance(
        result_test_df, pd.DataFrame
    ), "Expected test dataframe to be a pandas DataFrame"
    assert isinstance(
        result_test_labels_df, pd.DataFrame
    ), "Expected test labels dataframe to be a pandas DataFrame"

    pd.testing.assert_frame_equal(result_train_df, mock_train_df, check_exact=True)
    pd.testing.assert_frame_equal(result_test_df, mock_test_df, check_exact=True)
    pd.testing.assert_frame_equal(
        result_test_labels_df, mock_test_labels_df, check_exact=True
    )


@patch("src.data.download.kagglehub.load_dataset")
def test_load_data_invalid_handle(mock_load_dataset):
    invalid_handle = "invalid_handle"
    mock_load_dataset.side_effect = ValueError(
        f"Invalid dataset handle: {invalid_handle}"
    )
    with pytest.raises(ValueError, match=f"Invalid dataset handle: {invalid_handle}"):
        _ = load_data(handle=invalid_handle)


@patch("src.data.download.kagglehub.load_dataset")
def test_load_train_invalid_path(mock_load_dataset):
    mock_load_dataset.side_effect = KaggleApiHTTPError("404 Client Error")
    with pytest.raises(ValueError, match="Error loading train data: 404 Client Error"):
        _ = _load_train(handle="valid_handle")


@patch("src.data.download.kagglehub.load_dataset")
def test_load_test_invalid_path(mock_load_dataset):
    mock_load_dataset.side_effect = KaggleApiHTTPError("404 Client Error")
    with pytest.raises(ValueError, match="Error loading test data: 404 Client Error"):
        _ = _load_test(handle="valid_handle")


@patch("src.data.download.kagglehub.load_dataset")
def test_load_test_labels_invalid_path(mock_load_dataset):
    mock_load_dataset.side_effect = KaggleApiHTTPError("404 Client Error")
    with pytest.raises(
        ValueError, match="Error loading test labels data: 404 Client Error"
    ):
        _ = _load_test_labels(handle="valid_handle")
