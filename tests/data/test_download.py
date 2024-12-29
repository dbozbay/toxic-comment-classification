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
    mock_data = make_mock_data()
    mock_load_dataset.side_effect = mock_data

    handle = "valid_handle"
    result_data = load_data(handle)

    # Assertions to check if the dataframes are loaded correctly
    assert isinstance(result_data, tuple), "Expected to return a tuple"
    assert (
        len(result_data) == 3
    ), "Expected tuple to have exactly 3 elements (train, test, test_labels)"
    for result_df, mock_df in zip(result_data, mock_data):
        pd.testing.assert_frame_equal(result_df, mock_df, check_exact=True)


@patch("src.data.download.kagglehub.load_dataset")
def test_load_data_invalid_handle(mock_load_dataset):
    invalid_handle = "sooo_wrong_bro"
    mock_load_dataset.side_effect = ValueError(
        f"Invalid dataset handle: {invalid_handle}"
    )
    with pytest.raises(ValueError, match=f"Invalid dataset handle: {invalid_handle}"):
        _ = load_data(handle=invalid_handle)


@patch("src.data.download.kagglehub.load_dataset")
def test_load_train_invalid_path(mock_load_dataset):
    mock_load_dataset.side_effect = KaggleApiHTTPError("404 Client Error")
    handle = "valid_handle"
    with pytest.raises(ValueError, match="Error loading train data: 404 Client Error"):
        _ = _load_train(handle)


@patch("src.data.download.kagglehub.load_dataset")
def test_load_test_invalid_path(mock_load_dataset):
    mock_load_dataset.side_effect = KaggleApiHTTPError("404 Client Error")
    handle = "valid_handle"
    with pytest.raises(ValueError, match="Error loading test data: 404 Client Error"):
        _ = _load_test(handle)


@patch("src.data.download.kagglehub.load_dataset")
def test_load_test_labels_invalid_path(mock_load_dataset):
    mock_load_dataset.side_effect = KaggleApiHTTPError("404 Client Error")
    handle = "valid_handle"
    with pytest.raises(
        ValueError, match="Error loading test labels data: 404 Client Error"
    ):
        _ = _load_test_labels(handle)
