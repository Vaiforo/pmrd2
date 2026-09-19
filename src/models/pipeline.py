"""Полный конвейер обработки категориальных и числовых признаков."""

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.prepare import RANDOM_STATE
from src.features.rare_odor import RareOdorCounter


def build_pipeline() -> Pipeline:
    """Создать Pipeline; все статистики будут обучены только при fit.

    Числовая ветвь обрабатывает добавленный odor_rare_count.
    Исходные 22 признака остаются категориальными.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("rare_odor", RareOdorCounter()),
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
