"""Сохранение графиков качества без графического окружения."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import RocCurveDisplay  # noqa: E402


def save_evaluation_plots(
    target: pd.Series,
    probability: np.ndarray,
    matrix: np.ndarray,
    output_dir: Path,
) -> None:
    """Построить ROC и матрицу ошибок отложенной тестовой выборки."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(
        target, probability, ax=axes[0], name="Logistic regression"
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("ROC на тестовой выборке")
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["e: съедобный", "p: ядовитый"],
        yticklabels=["e: съедобный", "p: ядовитый"],
        ax=axes[1],
    )
    axes[1].set_title("Матрица ошибок на тесте")
    axes[1].set_xlabel("Предсказанный класс")
    axes[1].set_ylabel("Истинный класс")
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation.png", dpi=160)
    plt.close(figure)
