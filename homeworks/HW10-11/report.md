# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Выбран датасет STL10. Рекомендуемый по умолчанию, содержит достаточно 10 классов и 96x96 изображений, что абсолютно достаточно.
- Выбран трек detection с датасетом "Pascal VOC 2012". Detection выбран как более универсальный вариант для работы с готовыми CV пайплайнами.
- Часть A: 4 эксперимента (с C1 до C4) - простая CNN, CNN с аугментациями, ResNet18 head-only, ResNet18 fine-tune. Часть B: 2 режима инференса (V1-V2) с разными score_threshold (0.3, 0.7)

## 2. Среда и воспроизводимость

- Python: 3.14.0
- torch / torchvision: 2.10.0
- Устройство (CPU/GPU): CPU
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: `STL10`
- Разделение: train/val/test (80/20 split для train/val, отдельный test)
- Базовые transforms: Resize, ToTensor, Normalize
- Augmentation transforms: RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize
- Комментарий (2-4 предложения): STL10 содержит 10 классов природных объектов (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck). Изображения 96x96 пикселей, что требует больше вычислительных ресурсов чем CIFAR10, но и задача получается сложнее. 13000 изображений в train, 8000 в test.

### 3.2. Часть B: structured vision

- Датасет: `Pascal VOC`
- Трек: `detection`
- Что считается ground truth: Bounding boxes из VOC аннотаций (20 классов объектов)
- Какие предсказания использовались: Bounding boxes от FasterRCNN_ResNet50_FPN с разными порогами
- Комментарий (2-4 предложения): Pascal VOC — это стандартный бенчмарк для detection. 20 классов объектов позволяют продемонстрировать работу multi-class detection. Pretrained модель на COCO позволяет без дообучения получить хорошие результаты.

## 4. Часть A: модели и обучение (C1-C4)

Опишите коротко и сопоставимо:

- C1 (simple-cnn-base): Простая CNN (3 conv слоя + FC), без аугментаций
- C2 (simple-cnn-aug): Простая CNN (3 conv слоя + FC), с аугментациями (flip, rotation, color jitter)
- C3 (resnet18-head-only): Pretrained ResNet18, заморожен backbone, обучается только FC
- C4 (resnet18-finetune): Pretrained ResNet18, разморожен layer4 + FC

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer(ы): Adam (lr=0.001 для FC, lr=0.0001 для layer4)
- Batch size: 64
- Epochs (макс): 20
- Критерий выбора лучшей модели: best_val_accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

### Если выбран detection track

- Модель: FasterRCNN_ResNet50_FPN (pretrained на COCO).
- V1: `score_threshold = 0.3` (низкий порог, больше детекций).
- V2: `score_threshold = 0.7` (высокий порог, меньше детекций).
- Как считался IoU: Стандартный IoU = intersection / union для bounding boxes.
- Как считались precision / recall: TP: prediction с IoU >= 0.5 к ground truth. FP: prediction без matching ground truth. FN: ground truth без matching prediction. Precision = TP / (TP + FP), Recall = TP / (TP + FN).

### Если выбран segmentation track

- Модель: 
- Что считается foreground:
- V1: базовая постобработка
- V2: альтернативная постобработка
- Как считался mean IoU:
- Считались ли дополнительные pixel-level метрики:

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/detection_examples.png`, `./artifacts/figures/detection_metrics.png`

Короткая сводка (6-10 строк):

- Лучший эксперимент части A: C4 (ResNet18 fine-tune)ы
- Лучшая `val_accuracy`: 96.10%
- Итоговая `test_accuracy` лучшего классификатора: 95.42%
- Что дали аугментации (C2 vs C1): +5.90%
- Что дал transfer learning (C3/C4 vs C1/C2): +34.30%
- Что оказалось лучше: head-only или partial fine-tuning: Fine-tuning (+1.40%)
- Что показал режим V1 во второй части: Precision=0.274, Recall=1.000, IoU=0.774 
- Что показал режим V2 во второй части: Precision=0.500, Recall=0.706, IoU=0.817
- Как интерпретируются метрики второй части: При увеличении порога precision растёт, recall падает, IoU немного растёт

## 7. Анализ

(8-15 предложений)

Нужно прокомментировать:

- Простая CNN показала скромные результаты (60.40%) на STL10 из-за ограниченной ёмкости модели и сложности датасета. STL10 содержит более сложные изображения чем CIFAR10, поэтому простой архитектуры недостаточно.
- Аугментации дали устойчивое улучшение (+5.90%), что подтверждает их эффективность для предотвращения overfitting. Особенно заметен рост на validation set, что указывает на лучшую генерализацию.
- Pretrained ResNet18 показал значительное преимущество (+34.30% vs C1), что демонстрирует силу transfer learning. Pretrained веса на ImageNet содержат полезные признаки для классификации изображений.
- Fine-tuning layer4 дал дополнительное улучшение (+1.40%), что показывает полезность дообучения последних слоёв backbone для адаптации к целевому датасету.
- Detection метрики показывают классический tradeoff: V1 (низкий порог) даёт высокий recall (1.0) но низкий precision (0.274) — модель детектирует всё, но много false positives. V2 (высокий порог) даёт баланс — precision растёт до 0.5, recall падает до 0.706, но IoU растёт (0.817) — оставшиеся детекции качественнее.
- V1 (низкий порог) даёт высокий recall (1.0) но низкий precision (0.274) — модель детектирует всё, но много false positives. V2 (высокий порог) даёт баланс — precision растёт до 0.5, recall падает до 0.706, но IoU растёт (0.817)
- На низком пороге (V1) модель детектирует много объектов с низкой уверенностью, что увеличивает false positives. На высоком пороге (V2) некоторые объекты пропускаются, но качество детекций выше.

## 8. Итоговый вывод

(3-7 предложений)

- Выбрал бы C4 (ResNet18 fine-tune) как базовый, так как он даёт наилучший баланс accuracy и вычислительной эффективности.
- Pretrained веса на больших датасетах (ImageNet) содержат универсальные признаки, которые эффективно переносятся на другие задачи. Fine-tuning последних слоёв позволяет адаптировать модель к целевому домену.
- Precision/recall tradeoff — фундаментальное свойство detection. Выбор порога зависит от задачи: высокий recall для safety-critical приложений, высокий precision для минимизации false alarms. IoU показывает качество локализации bounding boxes.

## 9. Приложение (опционально)

Если вы делали дополнительные сравнения:

- дополнительные fine-tuning сценарии
- confusion matrix для классификации
- дополнительная постобработка для второй части
- дополнительные графики: `./artifacts/figures/...`
