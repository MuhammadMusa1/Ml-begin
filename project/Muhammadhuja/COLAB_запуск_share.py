# === ЗАПУСК ПРИЛОЖЕНИЯ ПРЯМО ИЗ COLAB (запасной вариант для защиты) ===
# Даёт публичную ссылку на ~72 часа. Не зависит от квоты Hugging Face.
# Сначала должна быть выполнена ячейка, где обучена и сохранена модель (mathe_model.pkl).

!pip install gradio -q

import gradio as gr
import pandas as pd
import joblib

model = joblib.load("mathe_model.pkl")

prep = model.named_steps["prep"]
_, transformer, features = prep.transformers_[0]
options = {c: [str(v) for v in cats] for c, cats in zip(features, transformer.categories_)}

def predict(*values):
    row = pd.DataFrame([dict(zip(features, values))])[features]
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    p_correct = proba[list(model.classes_).index(1)]
    verdict = "ВЕРНО" if pred == 1 else "НЕВЕРНО"
    return f"Скорее всего студент ответит: {verdict}\n\nУверенность в правильном ответе: {round(p_correct*100)}%"

inputs = [gr.Dropdown(options[c], label=c, value=options[c][0]) for c in features]
demo = gr.Interface(fn=predict, inputs=inputs, outputs=gr.Textbox(label="Прогноз", lines=3),
    title="Предсказание правильности ответа (MathE)",
    description="Модель по теме, уровню и стране студента оценивает, ответит ли он верно.")
demo.launch(share=True)
