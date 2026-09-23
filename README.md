# Лабораторная работа 1 — Mushroom

Вариант 16: классификация съедобности грибов и воспроизводимый конвейер на `scikit-learn`. Источник — [Mushroom, OpenML ID 24](https://www.openml.org/d/24), версия 1: 8124 строки, 22 категориальных признака и цель `class`. Класс `e` означает съедобный гриб, `p` — ядовитый.

**Все шаги лабораторной выполняются в Jupyter ноутбуке `notebooks/lab1_mushroom.ipynb`**. Ноутбук самодостаточен и не зависит от модулей `src/`. Скачивание данных, обучение и формирование результатов выполняет пользователь последовательно в ячейках ноутбука.

## Структура проекта

| Путь | Назначение |
| --- | --- |
| `notebooks/lab1_mushroom.ipynb` | **Основной ноутбук** — все шаги лабораторной (28 ячеек) |
| `data/raw/` | Исходный CSV; появится после выполнения ячейки загрузки |
| `data/processed/` | Train/test с индексом `source_row`; появятся после обучения |
| `models/` | Обученная модель после выполнения ноутбука |
| `reports/` | Шаблон отчёта; результаты создаются при выполнении ноутбука |
| `src/` | Зарезервирована для будущих скриптов (Lab 2+); содержит только .gitkeep |
| `tests/test_notebook.py` | Валидация структуры и логики ноутбука |

## Окружение

Все команды выполняйте из корня репозитория на ветке `lab1`. Используйте Python 3.12; версии библиотек зафиксированы.

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

В Windows: `py -3.12 -m venv venv`, затем в PowerShell `venv\Scripts\Activate.ps1`.

## Выполнение лабораторной

Откройте ноутбук в Jupyter:

```bash
jupyter lab notebooks/lab1_mushroom.ipynb
```

Последовательно выполните все ячейки ноутбука:

1. **Константы и конфигурация** — параметры эксперимента (OPENML_DATA_ID=24, RANDOM_STATE=42)
2. **Загрузка датасета** — скачивание OpenML ID 24 с проверкой SHA-256
3. **Валидация данных** — проверка формата, размера и пропусков
4. **Разбиение train/test** — 80/20, random_state=42, стратификация по class
5. **Анализ частот odor** — определение редких символов (< 10% строго)
6. **RareOdorCounter** — собственный трансформер (BaseEstimator + TransformerMixin)
7. **Pipeline** — числовая и категориальная ветви, LogisticRegression
8. **GridSearchCV** — подбор C из [0.1, 1.0, 10.0] на 5 фолдах по ROC-AUC
9. **Оценка test** — accuracy, ROC-AUC, F1, матрица ошибок
10. **Сохранение** — модель, метрики, разбиения, графики в `reports/` и `models/`
11. **Визуализация** — ROC-кривая и confusion matrix
12. **Итоговая сводка** — текстовое резюме в training_summary.txt
13. **DVC инструкции** — команды для фиксации данных
14. **Заполнение отчёта** — метрики и скриншоты

Все функции, классы и код определены непосредственно в ячейках ноутбука. Импорты из `src/` отсутствуют.

## DVC: фиксация данных и модели

После выполнения ноутбука зафиксируйте данные и модель через DVC:

```bash
# Добавить исходные данные
dvc add data/raw/mushroom.csv

# Добавить разбиения и модель
dvc add data/processed/train.csv data/processed/test.csv models/mushroom.joblib

# Проверить статус
dvc status

# Добавить .dvc файлы в Git
git add data/raw/mushroom.csv.dvc data/raw/.gitignore
git add data/processed/train.csv.dvc data/processed/test.csv.dvc data/processed/.gitignore
git add models/mushroom.joblib.dvc models/.gitignore

# Зафиксировать в Git
git commit -m "Track lab1 data and model with DVC"
```

DVC-кэш локальный. Для совместной работы настройте удалённое хранилище (`dvc remote add`) и выполните `dvc push`.

## Проверки

Валидация структуры и логики ноутбука без выполнения сетевых запросов или обучения:

```bash
python -m pytest tests/test_notebook.py -v
```

**Тесты:**
- Структура: валидность JSON, наличие секций, отсутствие outputs
- Код: все ячейки компилируются, нет импортов из src/
- Константы: OPENML_DATA_ID=24, RANDOM_STATE=42, TEST_SIZE=0.2, RARE_THRESHOLD=0.10
- RareOdorCounter: порог 10% строгий, неизвестные символы, пропуски, клонирование, валидация
- split_dataset: воспроизводимость, отсутствие пересечений train/test
- build_pipeline: импутация, обработка неизвестных категорий
- download_dataset: отсутствующий файл вызывает fetch, существующий валидный пропускает fetch, неверная контрольная сумма отклоняется

## Требования задания и реализация

| Требование | Реализация |
| --- | --- |
| Структура, Git, venv | Каталоги проекта, `.gitignore`, Python 3.12 |
| Фиксация зависимостей | `requirements.txt` с точными версиями |
| Скачивание в `data/raw` | Ячейка загрузки в ноутбуке с SHA-256 проверкой |
| Контроль данных DVC | Команды `dvc add` после выполнения ноутбука |
| Разделение 80/20 | `train_test_split`, random_state=42, стратификация |
| Собственный трансформер | `RareOdorCounter` класс в ячейке ноутбука |
| Числовая ветвь | Медиана → StandardScaler для `odor_rare_count` |
| Категориальная ветвь | Мода → OneHotEncoder(handle_unknown="ignore") |
| Модель и подбор | LogisticRegression, GridSearchCV по ROC-AUC |
| Оценка и сохранение | Accuracy, ROC-AUC, F1, матрица, joblib, графики |
| Отчёт и скриншоты | `reports/report_template.md` заполнить после выполнения |

## Отчёт

Заполните `reports/report_template.md` реальными метриками из `reports/metrics.json` и `reports/run_metadata.json`. Добавьте скриншоты:

1. Структура проекта
2. Вывод `dvc status`
3. Результаты обучения и метрики из ноутбука
4. Интерфейс Jupyter

Сохраните скриншоты в `reports/screenshots/` и явно добавьте в Git:

```bash
git add reports/report_template.md
git add -f reports/metrics.json reports/run_metadata.json reports/cv_results.csv
git add -f reports/training_summary.txt reports/odor_frequencies.csv
git add -f reports/figures/evaluation.png reports/screenshots/
git commit -m "Document lab1 results and screenshots"
```

## Ключевые особенности

- **Самодостаточный ноутбук**: все функции и классы определены в ячейках, импорты из `src/` отсутствуют
- **Воспроизводимость**: фиксированные seed, версии, SHA-256, BLAS single-thread
- **Отсутствие утечки**: разбиение до обучения, частоты пересчитываются в каждом CV-фолде
- **Граница редкости**: строго меньше 10% (ровно 10% не считается редким)
- **Контроль данных**: DVC для версионирования данных и модели
- **Валидация**: тесты проверяют структуру и логику ноутбука без выполнения

## Примечания

- Ноутбук сохранён без выполненных ячеек (все outputs пусты)
- Скачивание данных, обучение, метрики и скриншоты создаются при выполнении ячеек пользователем
- Автоматического обучения через CI/CD нет
- Загрузка сохранённой модели `joblib.load()` в свежем ядре требует предварительного определения класса `RareOdorCounter` в пространстве имён `__main__`. Выполнение ячеек определения из ноутбука может вызвать побочные эффекты (загрузку данных, вычисления). Для независимой загрузки модели скопируйте только определение класса без выполняемого кода.
