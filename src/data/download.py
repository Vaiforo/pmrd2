"""Скачать Mushroom только по явной команде пользователя.

Запуск из корня проекта: ``python -m src.data.download``.
Импорт модуля не обращается к сети и не создаёт файлов.
"""

import hashlib
from pathlib import Path

from sklearn.datasets import fetch_openml

from src.data.prepare import (
    EXPECTED_RAW_SHA256,
    OPENML_DATA_ID,
    PROJECT_ROOT,
    RAW_PATH,
    load_dataset,
)


def download_dataset() -> Path:
    """Сохранить исходный CSV или проверить уже имеющуюся копию.

    Фиксированный OpenML ID исключает выбор другой версии по имени.
    SHA-256 проверяется до записи: неожиданный ответ не заменит данные.
    Существующий файл проверяется, но никогда не перезаписывается.
    """
    if RAW_PATH.exists():
        load_dataset()
        return RAW_PATH

    dataset = fetch_openml(
        data_id=OPENML_DATA_ID,
        as_frame=True,
        parser="auto",
        data_home=str(PROJECT_ROOT / ".cache" / "openml"),
    )
    # Явные кодировка и перевод строки обеспечивают одинаковый CSV.
    content = dataset.frame.to_csv(index=False, lineterminator="\n").encode(
        "utf-8"
    )
    checksum = hashlib.sha256(content).hexdigest()
    if checksum != EXPECTED_RAW_SHA256:
        raise ValueError(
            "Ответ OpenML отличается от ожидаемого снимка Mushroom. "
            "CSV не сохранён; проверьте версии зависимостей и источник."
        )

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RAW_PATH.with_suffix(".csv.tmp")
    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(RAW_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    load_dataset()
    return RAW_PATH


def main() -> None:
    """Вывести путь к сохранённому и проверенному датасету."""
    path = download_dataset()
    print(f"Датасет готов: {path}")
    print("Следующий шаг: python -m src.data.prepare")


if __name__ == "__main__":
    main()
