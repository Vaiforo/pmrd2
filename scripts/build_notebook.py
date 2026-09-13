"""Создать и выполнить обзорный ноутбук с результатами лабораторной."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Собрать обзор из выполняемых ячеек, не дублируя обучение."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Лабораторная работа 1 — Mushroom\n\n"
            "Вариант 16. Порог редкости — строго меньше 10% в train.\n\n"
            "Обучение: python -m src.models.train. Этот ноутбук "
            "показывает готовые результаты и вызывает настоящий CLI DVC."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import subprocess\n"
            "import sys\n"
            "import joblib\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            "root = Path.cwd()\n"
            "if root.name == 'notebooks':\n"
            "    root = root.parent\n"
            "if str(root) not in sys.path:\n"
            "    sys.path.insert(0, str(root))\n"
        ),
        nbformat.v4.new_markdown_cell("## Структура проекта"),
        nbformat.v4.new_code_cell(
            "for name in ['data/raw', 'data/processed', 'notebooks', "
            "'src', 'models', 'reports']:\n"
            "    print(name + '/')\n"
            "    for path in sorted((root / name).rglob('*')):\n"
            "        if path.is_file() and '__pycache__' not in path.parts:\n"
            "            print('   ', path.relative_to(root))\n"
        ),
        nbformat.v4.new_markdown_cell(
            "## Проверка DVC\n\n"
            "Старый minidataset.csv отсутствовал уже в исходном проекте. "
            "Вторая команда проверяет только три файла Mushroom."
        ),
        nbformat.v4.new_code_cell(
            "scopes = [[], ['data/raw/mushroom.csv.dvc', "
            "'data/processed/train.csv.dvc', "
            "'data/processed/test.csv.dvc']]\n"
            "for scope in scopes:\n"
            "    print('$ python -m dvc status ' + ' '.join(scope))\n"
            "    result = subprocess.run(\n"
            "        [sys.executable, '-m', 'dvc', 'status', *scope],\n"
            "        cwd=root, capture_output=True, text=True, check=True)\n"
            "    print(result.stdout)\n"
        ),
        nbformat.v4.new_markdown_cell(
            "## Метрики и подбор параметров\n\n"
            "Цель e — съедобный, p — ядовитый. Положительный класс — p."
        ),
        nbformat.v4.new_code_cell(
            "print((root / 'reports/training_summary.txt').read_text())\n"
            "display(pd.read_csv(root / 'reports/cv_results.csv'))\n"
        ),
        nbformat.v4.new_markdown_cell("## Сохранённый конвейер"),
        nbformat.v4.new_code_cell(
            "model = joblib.load(root / 'models/mushroom.joblib')\n"
            "display(model)\n"
        ),
        nbformat.v4.new_markdown_cell(
            "## Вывод\n\n"
            "Метрики относятся к фиксированному тестовому разбиению "
            "учебного набора. Прирост от одного только счётчика не "
            "измерялся: отдельное сравнение без него не проводилось."
        ),
    ]
    path = ROOT / "notebooks/lab1_review.ipynb"
    nbformat.write(notebook, path)
    NotebookClient(
        notebook,
        timeout=120,
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print("Notebook executed successfully.")


if __name__ == "__main__":
    main()
