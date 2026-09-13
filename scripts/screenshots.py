"""Снять настоящие окна терминала с выполненными командами.

Нужны xterm, ImageMagick, tree и X-сервер. В CI используется xvfb-run.
Изображения не рисуются по тексту: ImageMagick захватывает окно X11.
"""

import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def commands_for(section: str) -> list[list[str]]:
    """Вернуть фиксированные команды для каждого подтверждения."""
    python = sys.executable
    if section == "structure":
        return [
            [
                "tree", "-d", "-L", "3",
                "-I", "__pycache__|.git|.venv|.cache",
                "data", "notebooks", "src", "models", "reports",
            ],
            ["find", "src", "-name", "*.py"],
        ]
    if section == "dvc-status":
        return [
            [python, "-m", "dvc", "status"],
            [
                python, "-m", "dvc", "status",
                "data/raw/mushroom.csv.dvc",
                "data/processed/train.csv.dvc",
                "data/processed/test.csv.dvc",
            ],
        ]
    return [["cat", "reports/training_summary.txt"]]


def worker(section: str, ready_path: Path) -> None:
    """Выполнить команды внутри настоящего xterm и оставить окно открытым."""
    for command in commands_for(section):
        shown = ["python", *command[1:]] if (
            command[0] == sys.executable
        ) else command
        print("$ " + " ".join(shown), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=True)
        if result.returncode != 0:
            raise RuntimeError("Команда завершилась с ошибкой.")
        print(flush=True)
    ready_path.write_text("ready")
    input("Commands completed. Press Enter to close.")


def main() -> None:
    """Захватить три окна после завершения соответствующих команд."""
    output = ROOT / "reports" / "screenshots"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="lab1-screen-") as directory:
        for section in ("structure", "dvc-status", "metrics"):
            ready = Path(directory) / section
            process = subprocess.Popen(
                [
                    "xterm", "-fa", "DejaVu Sans Mono", "-fs", "12",
                    "-geometry", "122x43+0+0",
                    "-bg", "#0d1117", "-fg", "#e6edf3",
                    "-title", f"Mushroom Lab 1 - {section}",
                    "-e", sys.executable, "-u", str(Path(__file__).resolve()),
                    "--worker", section, str(ready),
                ],
                cwd=ROOT,
            )
            try:
                deadline = time.monotonic() + 30
                while not ready.exists():
                    if process.poll() is not None:
                        raise RuntimeError("Окно закрылось до выполнения.")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Команды не завершились за 30 с.")
                    time.sleep(0.1)
                # Даём X-серверу отрисовать уже полученный вывод.
                time.sleep(0.5)
                subprocess.run(
                    [
                        "import", "-window", "root",
                        str(output / f"lab1-{section}.png"),
                    ],
                    check=True,
                )
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(sys.argv[2], Path(sys.argv[3]))
    else:
        main()
