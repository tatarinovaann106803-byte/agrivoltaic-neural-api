# Agrivoltaic Neural API

API сервис для расчёта эффективности агривольтаических систем (совместное размещение солнечных панелей и сельскохозяйственного производства).

Поддерживаемые сектора: растениеводство, аквакультура, лесное хозяйство.

## Возможности

- Автоматическое получение солнечной радиации и температуры из NASA POWER API
- Автоматическая подстановка цен на продукцию из базы Росстата
- Автоматическая подстановка базовой урожайности из FAO
- AI-предсказание изменения продуктивности (нейросеть или Random Forest)
- Расчёт энергетики
- Экономический расчёт (CAPEX, OPEX, срок окупаемости)

## Быстрый старт

Выполните команды по порядку:

```bash
# 1. Клонирование репозитория
git clone https://github.com/tatarinovaann106803-byte/agrivoltaic-neural-api.git
cd agrivoltaic-neural-api

# 2. Создание виртуального окружения
python -m venv venv
source venv/bin/activate        # для Mac/Linux
venv\Scripts\activate           # для Windows

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Скачивание и подготовка данных
python download_data.py
python prepare_dataset_advanced.py

# 5. Обучение моделей
python train_advanced_models.py
python smart_train.py

# 6. Запуск сервера
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
