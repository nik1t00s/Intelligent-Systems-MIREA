# Тесты проекта

Тесты находятся в файле:

```text
tests/test_service.py
```

Они проверяют:

- `GET /health`;
- успешный `POST /predict` на валидном JSON;
- ошибку валидации для некорректного запроса.

Команда запуска:

```bash
cd project
python -m unittest tests/test_service.py
```

Перед запуском тестов должна существовать сохраненная модель:

```text
models/model.joblib
models/model_metadata.json
```

Если артефактов нет, сначала выполните:

```bash
python -m src.train
```