"""Добавление числового счётчика редких кодов запаха Mushroom."""

from numbers import Real

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

RARE_THRESHOLD = 0.10
OUTPUT_COLUMN = "odor_rare_count"


class RareOdorCounter(TransformerMixin, BaseEstimator):
    """Считать редкие символы в однобуквенном признаке odor.

    Parameters
    ----------
    threshold : float, default=0.10
        Код редкий, если доля обучающих строк с ним строго меньше порога.
        Частота ровно 10% при пороге 0.10 не считается редкой.

    Notes
    -----
    Каждый odor в Mushroom содержит одну букву, поэтому счётчик равен
    0 или 1. Неизвестный код имеет частоту 0 и получает 1. Пропуск остаётся
    NaN для последующей импутации. Исходный odor сохраняется.
    Частоты обучаются только в fit, в том числе внутри каждого CV-фолда.
    """

    def __init__(self, threshold: float = RARE_THRESHOLD) -> None:
        self.threshold = threshold

    @staticmethod
    def _validate_frame(features: pd.DataFrame) -> None:
        """Проверить формат входа, не меняя переданный DataFrame."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("RareOdorCounter ожидает pandas.DataFrame.")
        if not features.columns.is_unique or "odor" not in features:
            raise ValueError("Нужны уникальные колонки и признак odor.")
        if OUTPUT_COLUMN in features:
            raise ValueError(f"Признак {OUTPUT_COLUMN} уже существует.")
        valid = (
            features["odor"]
            .dropna()
            .map(lambda value: isinstance(value, str) and len(value) == 1)
        )
        if not valid.all():
            raise ValueError("odor должен содержать однобуквенные коды.")

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "RareOdorCounter":
        """Запомнить частоты train; цель y не используется."""
        self._validate_frame(X)
        if len(X) == 0:
            raise ValueError("Обучающая выборка не должна быть пустой.")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, Real)
            or not 0 < self.threshold <= 1
        ):
            raise ValueError("threshold должен быть в интервале (0, 1].")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.frequencies_ = X["odor"].value_counts().div(len(X)).to_dict()
        self.rare_symbols_ = tuple(
            sorted(
                symbol
                for symbol, frequency in self.frequencies_.items()
                if frequency < self.threshold
            )
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Добавить счётчик по частотам fit, сохранив индексы и исходный X."""
        check_is_fitted(self, "frequencies_")
        self._validate_frame(X)
        if not np.array_equal(X.columns, self.feature_names_in_):
            raise ValueError("Колонки и их порядок должны совпадать с fit.")
        result = X.copy()
        odor = X["odor"]
        frequencies = odor.map(self.frequencies_).fillna(0.0)
        counts = (frequencies < self.threshold).astype(float)
        result[OUTPUT_COLUMN] = counts.mask(odor.isna(), np.nan)
        return result

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Вернуть имена исходных колонок и добавленного счётчика."""
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("input_features не совпадает с колонками fit.")
        return np.append(self.feature_names_in_, OUTPUT_COLUMN)
