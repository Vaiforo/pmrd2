"""Загрузить Mushroom и показать частоты кодов запаха только в train.

Запуск из корня репозитория: ``python -m src.data.prepare``.
Скрипт не назначает редкие категории: их выбирают после просмотра отчёта.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "mushroom.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "odor_frequencies.csv"
OPENML_DATA_ID = 24
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "class"

# Расшифровка кодов из описания Mushroom в OpenML и UCI.
ODOR_NAMES = {
    "a": "Миндальный",
    "l": "Анисовый",
    "c": "Креозотовый",
    "y": "Рыбный",
    "f": "Зловонный",
    "m": "Затхлый",
    "n": "Без запаха",
    "p": "Резкий",
    "s": "Пряный",
}


def load_dataset() -> pd.DataFrame:
    """Загрузить фиксированную версию OpenML либо прочитать локальный CSV.

    ID 24 однозначно задаёт Mushroom версии 1. CSV хранит исходные признаки
    без импутации и кодирования. Пропуски обозначаются пустыми ячейками;
    при чтении также распознаётся исходный маркер ``?``.
    """
    if not RAW_PATH.exists():
        dataset = fetch_openml(
            data_id=OPENML_DATA_ID,
            as_frame=True,
            parser="auto",
            data_home=str(PROJECT_ROOT / ".cache" / "openml"),
        )
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.frame.to_csv(RAW_PATH, index=False, lineterminator="\n")

    # dtype=object сохраняет номинальный характер всех исходных признаков.
    frame = pd.read_csv(RAW_PATH, dtype=object, na_values=["?"])
    if frame.shape != (8124, 23):
        raise ValueError(
            "Для OpenML ID 24 ожидается 8124 строки и 23 колонки."
        )
    if set(frame[TARGET_COLUMN].dropna()) != {"e", "p"}:
        raise ValueError("Целевая переменная должна содержать классы e и p.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Целевая переменная содержит пропуски.")
    return frame


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Разделить данные 80/20, сохранив пропорции классов и исходные индексы.

    Общая функция разбиения нужна, чтобы частоты категорий и последующее
    обучение использовали одну и ту же обучающую выборку.
    """
    features = frame.drop(columns=TARGET_COLUMN)
    target = frame[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def describe_odor(features_train: pd.DataFrame) -> pd.DataFrame:
    """Посчитать все коды odor в train, от самого редкого к частому.

    Доля считается от общего числа обучающих строк. Пропуски, если они
    появятся, выводятся отдельной строкой и не считаются кодом запаха.
    """
    counts = features_train["odor"].value_counts(dropna=False)
    report = counts.rename_axis("code").reset_index(name="train_count")
    report["meaning"] = report["code"].map(ODOR_NAMES).fillna("Пропуск")
    report["train_percent"] = 100 * report["train_count"] / len(features_train)
    return report.sort_values(["train_count", "code"]).reset_index(drop=True)


def main() -> None:
    """Сохранить таблицу частот и вывести результат для выбора категорий."""
    frame = load_dataset()
    features_train, features_test, _, _ = split_dataset(frame)
    report = describe_odor(features_train)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False, float_format="%.6f")

    print(f"OpenML ID: {OPENML_DATA_ID}; размер данных: {frame.shape}")
    print(f"Train: {len(features_train)}; test: {len(features_test)}")
    print(f"random_state={RANDOM_STATE}; stratify=class")
    print(
        report.to_string(
            index=False, float_format=lambda value: f"{value:.2f}"
        )
    )
    print("\nПропуски в исходных данных:")
    missing = frame.isna().sum()
    print(missing[missing > 0].to_string())
    print("\nРедкие коды пока не выбраны.")


if __name__ == "__main__":
    main()
