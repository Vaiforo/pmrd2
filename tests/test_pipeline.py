"""Проверка импутации, неизвестных категорий и сохранения Pipeline."""

import joblib
import numpy as np
import pandas as pd

from src.features.rare_odor import OUTPUT_COLUMN
from src.models.pipeline import build_pipeline


def test_pipeline_imputes_and_handles_unseen_categories(tmp_path):
    """Обе ветви обучаются на train; joblib сохраняет предсказания."""
    train = pd.DataFrame(
        {
            "odor": ["n"] * 12 + ["a"] * 7 + [np.nan],
            "stalk-root": ["b"] * 12 + ["r"] * 7 + [np.nan],
        }
    )
    model = build_pipeline().fit(train, pd.Series(["e"] * 12 + ["p"] * 8))
    preprocessor = model.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    assert numeric.named_steps["imputer"].statistics_.tolist() == [0.0]
    assert categorical.named_steps["imputer"].statistics_.tolist() == [
        "n",
        "b",
    ]
    assert f"numeric__{OUTPUT_COLUMN}" in model[:-1].get_feature_names_out()
    test = pd.DataFrame({"odor": ["z", np.nan], "stalk-root": ["q", np.nan]})
    prepared = model[:-1].transform(test)
    if hasattr(prepared, "toarray"):
        prepared = prepared.toarray()
    assert np.isfinite(prepared).all()
    assert prepared.shape[0] == 2
    original_probabilities = model.predict_proba(test)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_array_equal(
        original_probabilities, restored.predict_proba(test)
    )
