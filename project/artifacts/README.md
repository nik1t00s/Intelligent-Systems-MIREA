# Артефакты проекта

В этой папке лежат небольшие артефакты экспериментов:

```text
leaderboard.json
model_metadata.json
model.joblib
sample_tasks.csv
jira_issues_prepared.csv
```

Основная финальная модель для сервиса продублирована в каноническом пути:

```text
models/model.joblib
models/model_metadata.json
```

`leaderboard.json` содержит реальные метрики сравнения моделей. `sample_tasks.csv` содержит небольшой фрагмент подготовленных признаков для просмотра формата. `jira_issues_prepared.csv` генерируется командой:

```bash
python -m src.train
```

Большой исходный CSV с Kaggle не должен попадать в публичный репозиторий; инструкция получения данных находится в `data/README.md`.