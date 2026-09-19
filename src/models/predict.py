"""Получить прогноз сохранённого Pipeline для CSV с признаками."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data.prepare import PROJECT_ROOT


def main() -> None:
    """Показать первые прогнозы без повторного обучения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "mushroom.joblib",
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input_csv, dtype=object, na_values=["?"])
    model = joblib.load(args.model)
    expected_columns = model.feature_names_in_.tolist()
    missing_columns = sorted(set(expected_columns) - set(frame.columns))
    if missing_columns:
        parser.error(f"В CSV отсутствуют признаки: {missing_columns}")
    # Служебные class и source_row не передаются классификатору.
    features = frame.loc[:, expected_columns]
    poison_index = list(model.classes_).index("p")
    result = pd.DataFrame(
        {
            "predicted_class": model.predict(features),
            "probability_poisonous": model.predict_proba(features)[
                :, poison_index
            ],
        }
    )
    print(result.head(10).to_string(index=False))
    print(f"Всего обработано строк: {len(result)}")


if __name__ == "__main__":
    main()
