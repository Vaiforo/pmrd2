"""Повторить загрузку и обучение в копии без данных, модели и кэшей."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "data/raw/mushroom.csv",
    "data/processed/train.csv",
    "data/processed/test.csv",
    "models/mushroom.joblib",
    "reports/metrics.json",
    "reports/run_metadata.json",
    "reports/cv_results.csv",
    "reports/test_predictions.csv",
]


def digest(path: Path) -> str:
    """Вычислить SHA-256 фактических байтов файла."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Сравнить артефакты независимых запусков и DVC-указатели."""
    expected = {name: digest(ROOT / name) for name in FILES}
    with TemporaryDirectory(prefix="pmrd2-repro-") as directory:
        check_root = Path(directory)
        shutil.copytree(
            ROOT / "src",
            check_root / "src",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        # Новая копия не получает ни данные, ни OpenML/DVC-кэш.
        subprocess.run(
            [sys.executable, "-m", "src.models.train"],
            cwd=check_root,
            check=True,
        )
        matches = {
            name: {
                "match": digest(check_root / name) == checksum,
                "sha256": digest(check_root / name),
            }
            for name, checksum in expected.items()
        }
        if not all(record["match"] for record in matches.values()):
            raise RuntimeError(f"Невоспроизводимые артефакты: {matches}")
        subprocess.run(["git", "init", "-q"], cwd=check_root, check=True)
        subprocess.run(
            [sys.executable, "-m", "dvc", "init", "-q"],
            cwd=check_root,
            check=True,
        )
        for name in FILES[:3]:
            subprocess.run(
                [sys.executable, "-m", "dvc", "add", name],
                cwd=check_root,
                check=True,
            )
            if (
                (ROOT / (name + ".dvc")).read_bytes()
                != (check_root / (name + ".dvc")).read_bytes()
            ):
                raise RuntimeError(f"DVC-указатель изменился: {name}")
    result = {
        "method": "Fresh OpenML download and training without data or caches",
        "all_matched": True,
        "dvc_pointers_matched": True,
        "files": matches,
    }
    (ROOT / "reports/reproducibility.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print("Reproduction: all 8 files and 3 DVC pointers match.")


if __name__ == "__main__":
    main()
