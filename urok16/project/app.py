import spaces
import gradio as gr, tensorflow as tf, numpy as np
from tensorflow import keras

SIZE = 96
model = keras.models.load_model("model.keras")
class_names = ['квадрат', 'круг', 'треугольник']

@spaces.GPU
def predict(img):
    img = np.array(img)[..., :3]                 # PNG с прозрачностью -> RGB
    x = tf.image.resize(img, (SIZE, SIZE))[None, ...]
    p = model.predict(x, verbose=0)[0]
    return {class_names[i]: float(p[i]) for i in range(len(class_names))}

gr.Interface(fn=predict, inputs=gr.Image(),
             outputs=gr.Label(num_top_classes=len(class_names)),
             title="Классификатор фигур",
             description="Загрузи картинку с кругом, квадратом или треугольником.",
             flagging_mode="never").launch()
