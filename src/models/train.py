"""Обучить Pipeline, оценить отложенный тест и сохранить результаты.

Запуск из корня: python -m src.models.train.
Подбор параметров выполняется только на train.
"""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from src.data.prepare import (
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    describe_odor,
    load_dataset,
    split_dataset,
)
from src.models.pipeline import build_pipeline
from src.visualization.plots import save_evaluation_plots

MODEL_PATH = PROJECT_ROOT / "models" / "mushroom.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
C_VALUES = [0.1, 1.0, 10.0]
CV_FOLDS = 5


def write_json(path: Path, value: dict) -> None:
    """Записать JSON без NaN с единым форматированием."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def save_split(features: pd.DataFrame, target: pd.Series, name: str) -> None:
    """Сохранить split и исходные индексы для проверки разделения."""
    frame = features.copy()
    frame[TARGET_COLUMN] = target
    frame.to_csv(
        PROCESSED_DIR / f"{name}.csv",
        index=True,
        index_label="source_row",
        lineterminator="\n",
    )


def main() -> None:
    """Подобрать C на train и оценить выбранную модель на test."""
    for directory in (REPORTS_DIR, PROCESSED_DIR, MODEL_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
    frame = load_dataset()
    features_train, features_test, target_train, target_test = split_dataset(
        frame
    )
    save_split(features_train, target_train, "train")
    save_split(features_test, target_test, "test")
    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={"classifier__C": C_VALUES},
        scoring={"accuracy": "accuracy", "roc_auc": "roc_auc"},
        refit="roc_auc",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    # Один поток BLAS уменьшает различия округления между запусками.
    with threadpool_limits(limits=1):
        search.fit(features_train, target_train)
        model = search.best_estimator_
        predicted = model.predict(features_test)
        poison_index = list(model.classes_).index("p")
        poison_probability = model.predict_proba(features_test)[
            :, poison_index
        ]

    target_binary = (target_test == "p").astype(int)
    matrix = confusion_matrix(target_test, predicted, labels=["e", "p"])
    metrics = {
        "accuracy": float(accuracy_score(target_test, predicted)),
        "roc_auc": float(roc_auc_score(target_binary, poison_probability)),
        "f1_poisonous": float(f1_score(target_test, predicted, pos_label="p")),
        "test_rows": len(features_test),
        "errors": int(np.sum(predicted != target_test.to_numpy())),
        "confusion_matrix_labels": ["e", "p"],
        "confusion_matrix": matrix.tolist(),
    }
    write_json(REPORTS_DIR / "metrics.json", metrics)
    pd.DataFrame(
        {
            "source_row": features_test.index,
            "actual_class": target_test.to_numpy(),
            "predicted_class": predicted,
            "probability_poisonous": poison_probability,
        }
    ).to_csv(
        REPORTS_DIR / "test_predictions.csv", index=False, float_format="%.12f"
    )
    # Время выполнения не сохраняем: оно не воспроизводится побайтно.
    cv_results = pd.DataFrame(search.cv_results_)[
        [
            "param_classifier__C",
            "mean_test_accuracy",
            "std_test_accuracy",
            "mean_test_roc_auc",
            "std_test_roc_auc",
            "rank_test_roc_auc",
        ]
    ].rename(columns={"param_classifier__C": "C"})
    cv_results.to_csv(
        REPORTS_DIR / "cv_results.csv", index=False, float_format="%.12f"
    )
    rare_counter = model.named_steps["rare_odor"]
    feature_names = model[:-1].get_feature_names_out()
    describe_odor(features_train).to_csv(
        REPORTS_DIR / "odor_frequencies.csv",
        index=False,
        float_format="%.6f",
    )
    metadata = {
        "openml_id": 24,
        "openml_version": 1,
        "raw_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": len(features_train),
        "test_rows": len(features_test),
        "train_class_counts": target_train.value_counts().to_dict(),
        "test_class_counts": target_test.value_counts().to_dict(),
        "missing_values": frame.isna().sum().loc[lambda s: s > 0].to_dict(),
        "input_feature_count": features_train.shape[1],
        "transformed_feature_count": len(feature_names),
        "rare_threshold": rare_counter.threshold,
        "rare_comparison": "strictly_less_than",
        "rare_symbols": list(rare_counter.rare_symbols_),
        "odor_train_frequencies": rare_counter.frequencies_,
        "cv_folds": CV_FOLDS,
        "C_candidates": C_VALUES,
        "best_params": search.best_params_,
        "best_cv_roc_auc": float(search.best_score_),
        "positive_class": "p",
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "pandas",
                "numpy",
                "scipy",
                "scikit-learn",
                "joblib",
                "dvc",
            )
        },
    }
    write_json(REPORTS_DIR / "run_metadata.json", metadata)
    (REPORTS_DIR / "feature_names.json").write_text(
        json.dumps(feature_names.tolist(), indent=2) + "\n", encoding="utf-8"
    )
    # Сериализуем все стадии, включая статистики и кодировщик.
    joblib.dump(model, MODEL_PATH, compress=3)
    loaded_model = joblib.load(MODEL_PATH)
    with threadpool_limits(limits=1):
        np.testing.assert_array_equal(
            loaded_model.predict(features_test), predicted
        )
        np.testing.assert_allclose(
            loaded_model.predict_proba(features_test)[:, poison_index],
            poison_probability,
            rtol=0,
            atol=0,
        )
    save_evaluation_plots(
        target_binary, poison_probability, matrix, REPORTS_DIR / "figures"
    )
    summary = (
        f"OpenML Mushroom (ID 24), train={len(features_train)}, "
        f"test={len(features_test)}\n"
        f"Rare odor: frequency < {rare_counter.threshold:.0%}\n"
        f"Rare symbols: {', '.join(rare_counter.rare_symbols_)}\n"
        f"Best parameters: {search.best_params_}\n"
        f"Best mean CV ROC-AUC: {search.best_score_:.8f}\n"
        f"Test accuracy: {metrics['accuracy']:.8f}\n"
        f"Test ROC-AUC: {metrics['roc_auc']:.8f}\n"
        f"Test F1 (poisonous): {metrics['f1_poisonous']:.8f}\n"
        f"Errors: {metrics['errors']} / {len(features_test)}\n"
        f"Confusion matrix [e, p]: {matrix.tolist()}\n"
        "Model saved: models/mushroom.joblib\n"
        "Reload check: predictions and probabilities identical\n\n"
        + classification_report(target_test, predicted, digits=4)
    )
    (REPORTS_DIR / "training_summary.txt").write_text(
        summary, encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
