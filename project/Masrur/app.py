# app.py — Прогноз цены автомобиля
# Требуется файл модели рядом: car_model.pkl (обученный Pipeline из ноутбука forecastcar_исправленный)
# Интерфейс строится автоматически по признакам, зашитым в Pipeline.

import gradio as gr
import pandas as pd
import joblib

model = joblib.load("car_model.pkl")

# --- достаём имена признаков прямо из обученного Pipeline ---
prep = model.named_steps["prep"]  # ColumnTransformer
numeric_features, categorical_features = [], []
cat_options = {}

for name, transformer, cols in prep.transformers_:
    if name == "num":
        numeric_features = list(cols)
    elif name == "cat":
        categorical_features = list(cols)
        # OneHotEncoder хранит списки категорий -> используем как выпадающие значения
        try:
            onehot = transformer.named_steps["onehot"]
            for col, cats in zip(cols, onehot.categories_):
                cat_options[col] = [str(c) for c in cats]
        except Exception:
            for col in cols:
                cat_options[col] = []

features = numeric_features + categorical_features

def predict(*values):
    row = dict(zip(features, values))
    car = pd.DataFrame([row])[features]
    price = model.predict(car)[0]
    return f"Оценка: {round(price)} сомони"

# --- собираем поля ввода под конкретную модель ---
inputs = []
for col in numeric_features:
    inputs.append(gr.Number(label=col, value=0))
for col in categorical_features:
    opts = cat_options.get(col, [])
    inputs.append(gr.Dropdown(opts, label=col, value=(opts[0] if opts else None)))

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Прогноз цены"),
    title="Прогноз цены автомобиля",
    description="RandomForest в Pipeline. Введите параметры авто — модель оценит цену в сомони.",
)

demo.launch()
