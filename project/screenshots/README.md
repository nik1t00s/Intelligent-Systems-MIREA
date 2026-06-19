# Скриншоты

`swagger_ui.png` — подтверждение работы FastAPI Swagger UI.

Как обновить вручную:

1. Запустить сервис:

```bash
uvicorn src.service:app --host 0.0.0.0 --port 8000
```

2. Открыть:

```text
http://localhost:8000/docs
```

3. Сделать скриншот страницы, где видны endpoint-ы `/health` и `/predict`.
4. Сохранить файл как:

```text
screenshots/swagger_ui.png
```
