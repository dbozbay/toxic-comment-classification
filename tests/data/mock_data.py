from typing import Tuple

import pandas as pd


def make_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates a mock dataset with training, testing, and testing labels."""
    return _make_mock_train_df(), _make_mock_test_df(), _make_mock_test_labels()


def _make_mock_train_df() -> pd.DataFrame:
    """Creates a mock training dataset."""
    return pd.DataFrame(
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


def _make_mock_test_df() -> pd.DataFrame:
    """Creates a mock testing dataset (without labels)."""
    return pd.DataFrame(
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


def _make_mock_test_labels() -> pd.DataFrame:
    """Creates a mock testing dataset (with labels).
    In this case 3/5 samples are BAD, i.e. all labels are -1.
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "toxic": [-1, 0, -1, 0, -1],
            "severe_toxic": [-1, 1, -1, 0, -1],
            "obscene": [-1, 0, -1, 0, -1],
            "threat": [-1, 0, -1, 0, -1],
            "insult": [-1, 1, -1, 0, -1],
            "identity_hate": [-1, 0, -1, 1, -1],
        }
    )
