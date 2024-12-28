import os

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data():
    """Creates sample raw dataframes for testing"""
    train_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'comment_text': ['text1', 'text2', 'text3', 'text4', 'text5'],
        'toxic': [0, 1, 0, 1, 0],
        'severe_toxic': [0, 0, 1]
    })

    test_df = pd.DataFrame({
        'id': [6, 7, 6],
        'comment_text': ['text4', 'text5', 'text6']
    })

    test_labels_df = pd.DataFrame({
        'id': [4, 5, 6],
        'toxic': [1, 0, 0],
        'severe_toxic': [0, 0, 1]
    })

    return train_df, test_df, test_labels_df

# Sample data for testing
mock_train_df = pd.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "comment_text": ["comment1", "comment2", "comment3", "comment4", "comment5"],
        "toxic": [0, 1, 0, 1, 0],
        "severe_toxic": [0, 0, 0, 0, 0],
        "obscene": [0, 1, 0, 1, 0],
        "threat": [0, 0, 0, 0, 0],
        "insult": [0, 1, 0, 1, 0],
        "identity_hate": [0, 0, 0, 0, 0],
    }
)

mock_test_df = pd.DataFrame(
    {
        "id": [6, 7, 8, 9, 10],
        "comment_text": ["comment6", "comment7", "comment8", "comment9", "comment10"],
    }
)

mock_test_labels_df = pd.DataFrame(
    {
        "id": [6, 7, 8, 9, 10],
        "toxic": [-1, 0, 0, -1, 0],
        "severe_toxic": [-1, 0, 0, -1, 0],
        "obscene": [-1, 0, 0, -1, 0],
        "threat": [-1, 0, 0, -1, 0],
        "insult": [-1, 0, 0, -1, 0],
        "identity_hate": [-1, 0, 0, -1, 0],
    }
)

INPUT = ["comment_text"]
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def test_preprocess_data_default():
    train_df, val_df, test_df = preprocess_data(
        mock_train_df, mock_test_df, mock_test_labels_df, INPUT, LABELS
    )
    assert len(train_df) == 4
    assert len(val_df) == 1
    assert len(test_df) == 3
    assert train_df.shape[1] == len(INPUT) + len(LABELS)
    assert val_df.shape[1] == len(INPUT) + len(LABELS)
    assert test_df.shape[1] == len(INPUT) + len(LABELS)


def test_preprocess_data_test_size_0_3():
    train_df, val_df, test_df = preprocess_data(
        mock_train_df, mock_test_df, mock_test_labels_df, INPUT, LABELS, test_size=0.3
    )
    assert len(train_df) == 3
    assert len(val_df) == 2
    assert len(test_df) == 3


# def test_preprocess_data_random_state_1():
#     train_df, val_df, test_df = preprocess_data(mock_train_df, mock_test_df, mock_test_labels_df, INPUT, LABELS, random_state=1)
#     assert len(train_df) == 4
#     assert len(val_df) == 1
#     assert len(test_df) == 3


def test_preprocess_data_inputs_not_list():
    with pytest.raises(AssertionError):
        preprocess_data(
            mock_train_df, mock_test_df, mock_test_labels_df, "comment_text", LABELS
        )


def test_preprocess_data_labels_not_list():
    with pytest.raises(AssertionError):
        preprocess_data(
            mock_train_df, mock_test_df, mock_test_labels_df, INPUT, "toxic"
        )


def test_set_id_as_index():
    df = _set_id_as_index(mock_train_df)
    assert df.index.name == "id"


def test_merge_on_index():
    df1 = mock_test_df.set_index("id")
    df2 = mock_test_labels_df.set_index("id")
    merged_df = _merge_on_index(df1, df2)
    assert len(merged_df) == len(df1) == len(df2)


def test_remove_bad_test_samples():
    test_df = mock_test_df.set_index("id")
    test_labels_df = mock_test_labels_df.set_index("id")
    merged_df = _merge_on_index(test_df, test_labels_df)
    cleaned_df = _remove_bad_test_samples(merged_df, LABELS)
    assert len(cleaned_df) == 3


def test_split_train_validation():
    train_df, val_df = _split_train_validation(mock_train_df.set_index("id"))
    assert len(train_df) == 4
    assert len(val_df) == 1


def test_save_preprocessed_data(tmpdir):
    train_df, val_df, test_df = preprocess_data(
        mock_train_df, mock_test_df, mock_test_labels_df, INPUT, LABELS
    )
    output_dir = tmpdir.mkdir("processed_data")
    _save_preprocessed_data(train_df, val_df, test_df, output_dir=str(output_dir))
    assert os.path.exists(os.path.join(output_dir, "train.csv.gz"))
    assert os.path.exists(os.path.join(output_dir, "val.csv.gz"))
    assert os.path.exists(os.path.join(output_dir, "test.csv.gz"))

    loaded_train_df, loaded_val_df, loaded_test_df = load_preprocessed_data(
        input_dir=str(output_dir)
    )
    assert_frame_equal(train_df, loaded_train_df)
    assert_frame_equal(val_df, loaded_val_df)
    assert_frame_equal(test_df, loaded_test_df)
