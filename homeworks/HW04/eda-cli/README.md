# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Запуск HTTP API
```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

## Тесты

```bash
uv run pytest -q
```

## Примеры запросов к API

Проверка работоспособности:
```bash
curl http://localhost:8000/health
```

Анализ качества из CSV:
```bash
curl -X 'POST' 'http://localhost:8000/quality-from-csv' -H 'Content-Type: multipart/form-data' -F 'file=@data/example.csv'
```

Полный анализ флагов качества:

```bash
curl -X 'POST' 'http://localhost:8000/quality-flags-from-csv' -H 'Content-Type: multipart/form-data' -F 'file=@data/example.csv'
```

## HTTP API

- `/health` - GET, Проверка работоспособности сервиса

- `/quality` - POST, Анализ качества данных из JSON-данных (data (dict), n_rows (int), n_cols (int))

- `/quality-from-csv` - POST, Анализ качества данных из CSV-файла (file)

- `/quality-flags-from-csv` - POST, Полный анализ флагов качества из CSV-файла (file)