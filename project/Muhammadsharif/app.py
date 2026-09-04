# app.py — Прогноз времени на домашнее задание
# Space обучает модель при запуске (данные синтетические, обучение мгновенное).

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- данные (учебная демонстрация) ---
np.random.seed(42)
n = 100
subjects = np.random.randint(1, 7, size=n)
difficulty = np.random.randint(1, 6, size=n)
time_spent = subjects * 25 + difficulty * 15 + np.random.normal(0, 10, n)
time_spent = np.clip(time_spent, 15, 300)

X = pd.DataFrame({"subjects": subjects, "difficulty": difficulty})
y = time_spent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# baseline + модель
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
mae_baseline = mean_absolute_error(y_test, baseline.predict(X_test))

model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]).fit(X_train, y_train)
mae_model = mean_absolute_error(y_test, model.predict(X_test))

info = f"Baseline MAE: {mae_baseline:.1f} мин | Model MAE: {mae_model:.1f} мин"

def predict(subjects_count, difficulty_level):
    row = pd.DataFrame([[subjects_count, difficulty_level]], columns=["subjects", "difficulty"])
    pred = model.predict(row)[0]
    return f"{max(10, int(round(pred)))} минут"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(1, 6, step=1, value=3, label="Количество предметов"),
        gr.Slider(1, 5, step=1, value=3, label="Уровень сложности (1-5)"),
    ],
    outputs=gr.Textbox(label="Прогнозируемое время на ДЗ"),
    title="Прогноз времени на домашнее задание",
    description="Учебная демонстрация линейной регрессии. " + info,
)

demo.launch()
