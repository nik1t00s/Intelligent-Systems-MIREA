# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: (18000, 39)
- Целевая переменная: `target` (классы и их доли: 0 — 74%, 1 — 26%)
- Признаки: 39 числовых признаков (`id`, `f01`–`f35`, `x_int_1`, `x_int_2`, `target`), без категориальных

## 2. Protocol

- Разбиение: train/test, `test_size=0.25`, `random_state=666`, `stratify=y`
- Подбор: GridSearchCV на train, 5 фолдов, оптимизация ROC-AUC для моделей DecisionTree, RandomForest и GradientBoosting
- Метрики: accuracy, F1, ROC-AUC, что позволяет оценивать качество классификации и способность модели различать классы

## 3. Models

- DummyClassifier: стратегия `most_frequent`
- LogisticRegression: Pipeline(StandardScaler + LogisticRegression)
- DecisionTreeClassifier: подбор `max_depth` и `min_samples_leaf`
- RandomForestClassifier: подбор `max_depth`, `min_samples_leaf`, `max_features`
- GradientBoostingClassifier: подбор `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`
- StackingClassifier: базовые модели RandomForest и GradientBoosting, метамодель — LogisticRegression, обучение через CV-логику

## 4. Results

- Таблица/список финальных метрик на test по всем моделям

| Модель                      | accuracy | F1      | ROC-AUC |
|-----------------------------|----------|---------|---------|
| DummyClassifier             | 0.7373   | 0.0     | 0.5     |
| LogisticRegression          | 0.8138   | 0.568   | 0.8065  |
| DecisionTreeClassifier      | 0.8351   | 0.648   | 0.8377  |
| RandomForestClassifier      | 0.89     | 0.7506  | 0.9259  |
| GradientBoostingClassifier  | 0.8867   | 0.7539  | 0.9132  |
| StackingClassifier          | 0.9082    | 0.8106  | 0.9259   |

- Победитель: StackingClassifier — лучшая модель по accuracy, F1 и ROC-AUC, что подтверждает эффективность объединения ансамблей.

## 5. Analysis

- Устойчивость: изменение `random_state` в 1-2 моделях показывает небольшие колебания метрик (±0.01-0.02), что подтверждает стабильность моделей.
- Ошибки: матрица ошибок для StackingClassifier — TN:3202, FP:116, FN:298, TP: 884.
Модель хорошо различает классы, большинство ошибок приходится на меньший класс (target=1).
- Интерпретация: permutation importance (Top-10 признаков) для StackingClassifier — Наибольшее влияние оказывают признаки `f16` (0.07) и `f01` (0.034). Остальные признаки из топ-10 (`f08`, `f23`, `f07`, `f30`, `f12`, `f15`, `f13`, `f05`) имеют умеренное влияние (0.01–0.02). Модель в основном ориентируется на эти ключевые признаки, что соответствует ожидаемому распределению важности по данным.

## 6. Conclusion

- DummyClassifier показывает минимальные возможности различения классов.
- LogisticRegression значительно превосходит бейзлайн и демонстрирует эффективность признаков.
- Контроль сложности в DecisionTree и RandomForest помогают избежать переобучения и улучшить метрики.
- GradientBoosting показывает высокую точность, но StackingClassifier объединяет сильные стороны RandomForest и GradientBoosting, достигая лучших результатов.
- Метрики и графики (ROC, PR, матрица ошибок, permutation importance) подтверждают правильность построения модели и честный ML-протокол.
- Честный подход с CV и сохранением лучших моделей/метрик обеспечивает воспроизводимость эксперимента.