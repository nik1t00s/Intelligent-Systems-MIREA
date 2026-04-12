# Self-checklist

| # | Критерий | Да/Нет | Где смотреть / комментарий |
|---|---|---|---|
| 1 | Сервис запускается по инструкции из `README.md` и работает | ✅ | [README.md](/n:/AI/Intelligent-Systems-MIREA/project/README.md), разделы «Запуск локально» и «Docker» |
| 2 | `/predict` использует реальную обученную модель, а не заглушку | ✅ | [src/service.py](/n:/AI/Intelligent-Systems-MIREA/project/src/service.py), [artifacts/model.joblib](/n:/AI/Intelligent-Systems-MIREA/project/artifacts/model.joblib) |
| 3 | Есть EDA и хотя бы один notebook с экспериментами | ✅ | [notebooks/01_eda_and_experiments.ipynb](/n:/AI/Intelligent-Systems-MIREA/project/notebooks/01_eda_and_experiments.ipynb) |
| 4 | Есть сравнение нескольких моделей по метрикам | ✅ | [src/train.py](/n:/AI/Intelligent-Systems-MIREA/project/src/train.py), [report.md](/n:/AI/Intelligent-Systems-MIREA/project/report.md) |
| 5 | Код не свален в один notebook, есть структура в `src/` | ✅ | [src](/n:/AI/Intelligent-Systems-MIREA/project/src) |
| 6 | Есть Dockerfile и понятный сценарий развёртывания | ✅ | [Dockerfile](/n:/AI/Intelligent-Systems-MIREA/project/Dockerfile), [docker-compose.yml](/n:/AI/Intelligent-Systems-MIREA/project/docker-compose.yml), [README.md](/n:/AI/Intelligent-Systems-MIREA/project/README.md) |
| 7 | Есть `.env.example`, в репозитории нет реальных секретов | ✅ | [.env.example](/n:/AI/Intelligent-Systems-MIREA/project/.env.example) |
| 8 | Реализованы логи и базовая наблюдаемость | ✅ | [src/service.py](/n:/AI/Intelligent-Systems-MIREA/project/src/service.py), endpoints `/health` и `/metrics` |
| 9 | В `report.md` обоснован выбор финальной модели | ✅ | [report.md](/n:/AI/Intelligent-Systems-MIREA/project/report.md), раздел «Выбор финальной модели» |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | [README.md](/n:/AI/Intelligent-Systems-MIREA/project/README.md), [report.md](/n:/AI/Intelligent-Systems-MIREA/project/report.md) |

## Итог

Самооценка: `10/10` по чек-листу.  
Основание для максимального балла: проект покрывает полный минимально-производственный цикл от данных и экспериментов до сервиса, конфигов, тестов, контейнеризации и отчётной документации.
