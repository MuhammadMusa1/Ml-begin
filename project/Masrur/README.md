---
title: Прогноз цены авто
emoji: 🚗
colorFrom: purple
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Прогноз цены автомобиля

Регрессия (RandomForest в Pipeline): оценивает цену б/у авто по его параметрам.

## Что загрузить в Space
- `app.py`
- `requirements.txt`
- `car_model.pkl`  ← сохранить в ноутбуке: joblib.dump(model, "car_model.pkl")

Интерфейс строится автоматически по признакам, зашитым в обученный Pipeline.
