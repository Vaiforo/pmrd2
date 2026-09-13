"""Проверки разбиения до обучения преобразований."""

import pandas as pd

from src.data.prepare import describe_odor, split_dataset


def test_split_preserves_rows_and_target_alignment():
    """Train/test не пересекаются; цель соответствует исходным строкам."""
    frame = pd.DataFrame({"odor": ["n", "a"] * 50, "class": ["e", "p"] * 50})
    train, test, y_train, y_test = split_dataset(frame)
    again = split_dataset(frame)
    assert len(train) == 80 and len(test) == 20
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(frame.index)
    assert train.index.equals(y_train.index)
    assert test.index.equals(y_test.index)
    pd.testing.assert_frame_equal(train, again[0])
    assert "class" not in train.columns
    assert describe_odor(train)["train_count"].sum() == len(train)
