# Agrivoltaic Calculator API

API сервис для расчета эффективности агривольтаических систем.

## Структура проекта

- `app.py` - основной FastAPI сервер
- `train_advanced_models.py` - обучение Random Forest моделей
- `train_neural_models.py` - обучение нейросетевых моделей
- `predict_neural.py` - предсказания с использованием нейросетей
- `auto_train.py` - автоматическое переобучение при изменении данных
- `prepare_dataset_advanced.py` - подготовка датасетов
- `download_data.py` - скачивание данных
- `requirements.txt` - зависимости Python
- `runtime.txt` - версия Python
- `Procfile` - для деплоя на Render

## Установка

```bash
pip install -r requirements.txt
python prepare_dataset_advanced.py
python train_advanced_models.py
python train_neural_models.py
