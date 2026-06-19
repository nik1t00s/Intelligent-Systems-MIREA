# Jira Issue Delay Risk Service

Итоговый проект по программе ДПО «Инженерия искусственного интеллекта».

Сервис прогнозирует риск долгого закрытия Jira issue по признакам, доступным на этапе triage. Проект закрывает полный минимальный ML lifecycle: данные -> EDA -> предобработка -> baseline/improved models -> сохраненная модель -> FastAPI API -> Docker -> тесты -> артефакты для отчета.

## ML-задача

- Тип задачи: бинарная классификация.
- Входные данные: `issue_type`, `priority`, `has_priority`, `component_present`, `summary_length`, `summary_word_count`, `description_length`, `description_word_count`.
- Целевая переменная: `is_delayed`.
- Логика цели: `is_delayed = 1`, если `resolution_time_days = Resolved - Created` выше 75-го процентиля.
- Текущий порог: `305.2056` дня.
- Основные метрики: `F1`, `Recall`, `ROC-AUC`.
- Почему эти метрики: для раннего выявления рискованных issues важно не пропускать долгие задачи, поэтому кроме общего качества учитываются `Recall` и баланс `Precision/Recall` через `F1`.

## Структура

```text
project/
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── service.py
├── models/
│   ├── model.joblib
│   └── model_metadata.json
├── configs/
│   └── config.yaml
├── tests/
│   └── test_service.py
├── screenshots/
│   └── swagger_ui.png
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## Данные

Источник: Kaggle Jira Dataset.

```text
https://www.kaggle.com/datasets/cesaranasco/jira-dataset
```

Ожидаемый локальный путь:

```text
data/jira_dataset.csv
```

Инструкция по получению данных лежит в `data/README.md`. CSV не коммитится: правило добавлено в `.gitignore`.

## EDA

Открыть notebook:

```text
notebooks/01_eda.ipynb
```

В нем есть загрузка данных, описание структуры, проверка пропусков, типы признаков, базовая статистика, анализ целевой переменной, графики и выводы по дисбалансу, пропускам, выбросам и особенностям признаков.

## Обучение

```bash
cd project
pip install -r requirements.txt
python -m src.train
```

Скрипт:

- читает `data/jira_dataset.csv`;
- готовит `artifacts/jira_issues_prepared.csv`;
- строит единый sklearn pipeline с предобработкой из `src/preprocessing.py`;
- сравнивает baseline и улучшенные модели;
- сохраняет финальную модель в `models/model.joblib`;
- сохраняет метаданные в `models/model_metadata.json`.

## Modeling notebook

```text
notebooks/02_modeling.ipynb
```

Ноутбук загружает реальные метрики из `artifacts/leaderboard.json`, формирует таблицу сравнения моделей и строит график для отчета.

## Модели и метрики

| Модель | Описание | Accuracy | Precision | Recall | F1 | ROC-AUC | Комментарий |
|---|---|---:|---:|---:|---:|---:|---|
| LogisticRegression | Baseline, линейная модель | 0.5083 | 0.3112 | 0.8068 | 0.4491 | 0.6436 | Для сравнения |
| RandomForest | Improved, ансамбль деревьев | 0.6933 | 0.4465 | 0.9781 | 0.6131 | 0.9198 | Финальная модель |
| GradientBoosting | Improved, boosting деревьев | 0.7972 | 0.9364 | 0.1968 | 0.3253 | 0.8960 | Высокий precision, низкий recall |

Финальная модель: `RandomForest`. Она выбрана по лучшему `F1`, высокому `Recall` и высокому `ROC-AUC`.

## Локальный запуск API

```bash
cd project
pip install -r requirements.txt
uvicorn src.service:app --host 0.0.0.0 --port 8000
```

Альтернативно:

```bash
python -m src.service
```

Swagger UI:

```text
http://localhost:8000/docs
```

## API

Health-check:

```bash
curl http://localhost:8000/health
```

Пример ответа:

```json
{
  "status": "ok",
  "environment": "development",
  "model_loaded": true,
  "model_name": "random_forest",
  "model_version": "v1",
  "dataset": "data/jira_dataset.csv"
}
```

Predict:

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  --data @src/demo_request.json
```

Пример тела запроса:

```json
{
  "issues": [
    {
      "issue_type": "Bug",
      "priority": "High",
      "has_priority": true,
      "component_present": true,
      "summary_length": 92,
      "summary_word_count": 12,
      "description_length": 1840,
      "description_word_count": 245
    }
  ]
}
```

Пример ответа:

```json
{
  "model_name": "random_forest",
  "model_version": "v1",
  "predictions": [
    {
      "prediction": 0,
      "probability": 0.1699,
      "is_delayed": 0,
      "delay_probability": 0.1699,
      "delay_risk": "low"
    }
  ]
}
```

## Docker

```bash
cd project
docker build -t aie-project .
docker run -p 8000:8000 aie-project
```

Или через compose:

```bash
docker compose up --build
```

Контейнер запускает FastAPI на порту `8000`.

## Swagger UI screenshot

- URL: `http://localhost:8000/docs`
- Скриншот: `screenshots/swagger_ui.png`
- Инструкция по обновлению: `screenshots/README.md`

## Тесты

```bash
cd project
python -m unittest tests/test_service.py
```

Тесты проверяют:

- `GET /health`;
- успешный `POST /predict`;
- ошибку валидации для невалидного запроса.

## Конфигурация и безопасность

- Основной конфиг: `configs/config.yaml`.
- Пример окружения: `.env.example`.
- Реальный `.env` не коммитится.
- Секретов, токенов и паролей в проекте нет.
- В логах фиксируются запуск сервиса, загрузка модели, запросы, ошибки валидации и latency, без секретов и персональных данных.

## Сценарий демонстрации

1. Показать `data/README.md` и локальный `data/jira_dataset.csv`.
2. Открыть `notebooks/01_eda.ipynb`: EDA, пропуски, типы, статистика, целевая переменная, графики.
3. Открыть `notebooks/02_modeling.ipynb`: таблица сравнения моделей.
4. Запустить `python -m src.train` или показать уже сохраненную `models/model.joblib`.
5. Запустить API через `uvicorn src.service:app --host 0.0.0.0 --port 8000`.
6. Открыть `http://localhost:8000/docs` и показать `/health`, `/predict`.
7. Выполнить `curl http://localhost:8000/health`.
8. Выполнить `/predict` с `src/demo_request.json`.
9. Показать `screenshots/swagger_ui.png` и `tests/test_service.py`.

## Чек-лист

| Пункт | Статус | Где смотреть |
|---|---|---|
| Сервис запускается | Да | `src/service.py`, раздел «Локальный запуск API» |
| `/predict` использует реальную модель | Да | `src/service.py`, `models/model.joblib` |
| Есть EDA | Да | `notebooks/01_eda.ipynb` |
| Есть baseline и improved models | Да | `src/train.py`, `notebooks/02_modeling.ipynb` |
| Есть реальные метрики | Да | `artifacts/leaderboard.json`, таблица в README |
| Код структурирован в `src/` | Да | `src/` |
| Предобработка вынесена в код | Да | `src/preprocessing.py` |
| Есть Dockerfile | Да | `Dockerfile` |
| Есть `.env.example` | Да | `.env.example` |
| Секреты не коммитятся | Да | `.gitignore`, раздел «Конфигурация и безопасность» |
| Есть `/health` | Да | `src/service.py` |
| Есть логи | Да | `src/service.py` |
| Обоснован выбор модели | Да | раздел «Модели и метрики» |
| Есть Swagger screenshot | Да | `screenshots/swagger_ui.png` |
| Есть сохраненная модель | Да | `models/model.joblib` |
| Есть curl для `/predict` | Да | раздел «API» |
| Есть тесты | Да | `tests/test_service.py` |
