# app.py — Предсказание правильности ответа студента (MathE)
# Рядом должен лежать файл модели: mathe_model.pkl

import gradio as gr
import pandas as pd
import joblib

model = joblib.load("mathe_model.pkl")

# признаки и варианты для выпадающих списков достаём прямо из обученной модели
prep = model.named_steps["prep"]           # ColumnTransformer
name, transformer, features = prep.transformers_[0]  # ("cat", OneHotEncoder, [колонки])
options = {col: [str(v) for v in cats] for col, cats in zip(features, transformer.categories_)}

def predict(*values):
    row = pd.DataFrame([dict(zip(features, values))])[features]
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    # вероятность класса "верно" (1)
    classes = list(model.classes_)
    p_correct = proba[classes.index(1)] if 1 in classes else max(proba)
    verdict = "ВЕРНО ✅" if pred == 1 else "НЕВЕРНО ❌"
    return f"Скорее всего студент ответит: {verdict}\n\nУверенность в правильном ответе: {round(p_correct*100)}%"

inputs = [gr.Dropdown(options[col], label=col, value=options[col][0]) for col in features]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Прогноз", lines=3),
    title="Предсказание правильности ответа (MathE)",
    description="Модель по теме, уровню и стране студента оценивает, ответит ли он верно.",
)

demo.launch()
