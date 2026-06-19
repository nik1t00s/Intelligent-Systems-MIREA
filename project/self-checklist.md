# Self-checklist

| # | Критерий | Да/Нет | Где смотреть / комментарий |
|---|---|---|---|
| 1 | Сервис запускается по инструкции из `README.md` и работает | Да | `README.md`, разделы «Локальный запуск API» и «Docker» |
| 2 | `/predict` использует реальную обученную модель, а не заглушку | Да | `src/service.py`, `models/model.joblib` |
| 3 | Есть EDA и хотя бы один notebook/эксперимент с метриками | Да | `notebooks/01_eda.ipynb`, `notebooks/02_modeling.ipynb`, `artifacts/leaderboard.json` |
| 4 | Есть baseline и улучшенная модель, сравнение по метрикам | Да | `src/train.py`, `artifacts/leaderboard.json`, `notebooks/02_modeling.ipynb` |
| 5 | Код не свален в один notebook, есть структура в `src/` | Да | `src/preprocessing.py`, `src/train.py`, `src/predict.py`, `src/service.py`, `src/schemas.py` |
| 6 | Есть Dockerfile и понятный сценарий развёртывания | Да | `Dockerfile`, `docker-compose.yml`, `README.md` |
| 7 | Есть `.env.example`, в репозитории нет реальных секретов | Да | `.env.example`, `.gitignore` |
| 8 | Реализованы логи и базовая наблюдаемость | Да | `src/service.py`, endpoints `/health` и `/metrics` |
| 9 | Обоснован выбор финальной модели | Да | `report.md`, `README.md`, `artifacts/leaderboard.json` |
| 10 | `README.md` и отчет позволяют понять сценарий демонстрации | Да | `README.md`, `report.md`, `Шаблон отчета.docx` / `Шаблон отчета.pdf` |

## Итог

Самооценка: `10/10` по чек-листу.

Проект использует реальный Jira-датасет `data/jira_dataset.csv` с Kaggle, а не синтетические данные. Сам CSV не должен загружаться в публичный GitHub без проверки условий Kaggle; для проверки структуры в репозитории есть `data/demo_sample.csv`, а инструкция получения данных описана в `data/README.md`.

Целевая переменная `is_delayed` строится по фактическому времени между `Created` и `Resolved`; признаки с прямой утечкой будущей информации исключены из обучения.