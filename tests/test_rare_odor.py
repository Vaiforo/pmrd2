"""Проверки границы 10%, отсутствия утечки и контракта трансформера."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.features.rare_odor import OUTPUT_COLUMN, RareOdorCounter


def test_threshold_is_strict_and_does_not_learn_from_test():
    """10% не редкая частота, 5% редкая; test не меняет статистики."""
    train = pd.DataFrame({"odor": ["n"] * 17 + ["a"] * 2 + ["m"]})
    transformer = RareOdorCounter().fit(train)
    assert transformer.rare_symbols_ == ("m",)
    before = transformer.frequencies_.copy()
    test = pd.DataFrame({"odor": ["m"] * 8 + ["a", "n"]})
    result = transformer.transform(test)
    assert result[OUTPUT_COLUMN].tolist() == [1.0] * 8 + [0.0, 0.0]
    assert transformer.frequencies_ == before


def test_missing_unknown_values_and_input_preservation():
    """NaN не заменяется кодом, неизвестный код редкий, X не меняется."""
    transformer = RareOdorCounter().fit(pd.DataFrame({"odor": ["n"] * 20}))
    test = pd.DataFrame({"odor": ["q", np.nan, "n"]}, index=[10, 4, 99])
    original = test.copy(deep=True)
    result = transformer.transform(test)
    pd.testing.assert_frame_equal(test, original)
    assert result.index.tolist() == [10, 4, 99]
    assert result.loc[10, OUTPUT_COLUMN] == 1
    assert pd.isna(result.loc[4, OUTPUT_COLUMN])
    assert result.loc[99, OUTPUT_COLUMN] == 0
    assert transformer.get_feature_names_out().tolist() == [
        "odor",
        OUTPUT_COLUMN,
    ]


def test_estimator_can_be_cloned_without_fitted_state():
    """GridSearchCV получает независимые необученные копии."""
    trained = RareOdorCounter(0.2).fit(pd.DataFrame({"odor": ["n", "a"]}))
    fresh = clone(trained)
    assert fresh.threshold == 0.2
    assert not hasattr(fresh, "frequencies_")
    with pytest.raises(NotFittedError):
        fresh.transform(pd.DataFrame({"odor": ["a"]}))


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, True, "0.1"])
def test_invalid_threshold_rejected(threshold):
    """Ошибочный порог не превращается молча в другое правило."""
    with pytest.raises(ValueError, match="threshold"):
        RareOdorCounter(threshold).fit(pd.DataFrame({"odor": ["n"]}))


def test_invalid_schema_rejected():
    """Трансформер требует odor и однобуквенные коды."""
    with pytest.raises(ValueError, match="odor"):
        RareOdorCounter().fit(pd.DataFrame({"smell": ["n"]}))
    with pytest.raises(ValueError, match="однобуквенные"):
        RareOdorCounter().fit(pd.DataFrame({"odor": ["almond"]}))
