# Система управления задачами с прогнозированием просрочки

End-to-end учебный проект по курсу «Инженерия искусственного интеллекта».  
Предметная область: управление задачами в небольшой команде.  
ML-задача: бинарная классификация `is_overdue` — предсказать, будет ли задача просрочена.

## Что внутри

- Реалистичный синтетический датасет задач в [data/tasks_synthetic.csv](/n:/AI/Intelligent-Systems-MIREA/project/data/tasks_synthetic.csv)
- Обучение и сравнение `LogisticRegression`, `RandomForest` и `GradientBoosting`
- Сохранённая обученная модель в [artifacts/model.joblib](/n:/AI/Intelligent-Systems-MIREA/project/artifacts/model.joblib)
- FastAPI-сервис с `/health`, `/predict`, `/metrics`
- Базовая наблюдаемость: логи, счётчики запросов, latency, health-check
- Конфиги и переменные окружения через [configs/train_config.yaml](/n:/AI/Intelligent-Systems-MIREA/project/configs/train_config.yaml) и [.env.example](/n:/AI/Intelligent-Systems-MIREA/project/.env.example)
- Docker-упаковка и минимальные smoke-тесты

## Структура

- [src](/n:/AI/Intelligent-Systems-MIREA/project/src) — генерация данных, обучение, API-сервис
- [data](/n:/AI/Intelligent-Systems-MIREA/project/data) — синтетический датасет
- [artifacts](/n:/AI/Intelligent-Systems-MIREA/project/artifacts) — модель и метаданные экспериментов
- [notebooks](/n:/AI/Intelligent-Systems-MIREA/project/notebooks) — EDA и эксперименты
- [tests](/n:/AI/Intelligent-Systems-MIREA/project/tests) — проверки API
- [report.md](/n:/AI/Intelligent-Systems-MIREA/project/report.md) — краткий отчёт
- [self-checklist.md](/n:/AI/Intelligent-Systems-MIREA/project/self-checklist.md) — самопроверка

## Запуск локально

```bash
cd project
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
uvicorn src.service:app --host 0.0.0.0 --port 8000
```

Сервис будет доступен на `http://localhost:8000`.

## Эндпоинты

- `GET /health` — статус сервиса и имя загруженной модели
- `POST /predict` — прогноз риска просрочки по списку задач
- `GET /metrics` — базовые текстовые метрики в стиле Prometheus

## Пример запроса к API

Файл с примером тела запроса: [src/demo_request.json](/n:/AI/Intelligent-Systems-MIREA/project/src/demo_request.json)

```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  --data @src/demo_request.json
```

Пример ответа:

```json
{
  "model_name": "logistic_regression",
  "predictions": [
    {
      "is_overdue": 1,
      "overdue_probability": 0.9425,
      "overdue_risk": "high"
    },
    {
      "is_overdue": 0,
      "overdue_probability": 0.0853,
      "overdue_risk": "low"
    }
  ]
}
```

## Обучение модели

```bash
cd project
python -m src.train
```

Скрипт:

- генерирует синтетический датасет на 5000 задач;
- делит выборку на `train/test = 80/20`;
- обучает три модели;
- сохраняет финальную модель и таблицу метрик в `artifacts/`.

Фактические метрики на тестовой выборке:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LogisticRegression | 0.6700 | 0.4637 | 0.6622 | 0.5455 | 0.7270 |
| RandomForest | 0.7190 | 0.5421 | 0.3880 | 0.4522 | 0.6986 |
| GradientBoosting | 0.7320 | 0.6148 | 0.2776 | 0.3825 | 0.6946 |

Финальная модель: `LogisticRegression`, потому что при умеренно несбалансированных классах она лучше удерживает `recall` и даёт лучший `F1`, что важнее для раннего обнаружения потенциально просроченных задач.

## Тесты

```bash
cd project
python -m unittest tests/test_service.py
```

Тесты проверяют:

- запуск API и `/health`;
- рабочий сценарий `/predict`;
- валидацию входных данных.

## Docker

```bash
cd project
docker build -t task-overdue-service .
docker run -p 8000:8000 task-overdue-service
```

Или:

```bash
cd project
docker compose up --build
```

## Сценарий демонстрации

1. Показать структуру проекта и артефакты в `data/`, `artifacts/`, `notebooks/`, `src/`.
2. Открыть [notebooks/01_eda_and_experiments.ipynb](/n:/AI/Intelligent-Systems-MIREA/project/notebooks/01_eda_and_experiments.ipynb) и показать EDA с таблицей сравнения моделей.
3. Запустить сервис командой `uvicorn src.service:app --host 0.0.0.0 --port 8000`.
4. Проверить `GET /health`.
5. Отправить `POST /predict` с примером из `src/demo_request.json`.
6. Показать `GET /metrics` и объяснить, какие счётчики обновились.

## Ограничения

- Датасет синтетический, поэтому метрики отражают качество на искусственно сгенерированном процессе.
- В сервисе нет аутентификации и внешнего хранилища.
- Мониторинг реализован в минимальном учебном объёме.

