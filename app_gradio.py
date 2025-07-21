import gradio as gr
import tensorflow as tf
import numpy as np
import json
from PIL import Image

MODEL_PATH = "oxford_pets_production.h5"
META_PATH = "model_metadata.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)
class_names = metadata["class_names"]
IMG_SIZE = metadata["img_size"]

def predict(image):
    if image is None:
        return "<div style='color:#e74c3c;font-size:1.2em;'>Lütfen bir resim yükleyin.</div>", None
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]
    predicted_class = class_names[class_idx]
    # Top-3 tahmin
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(class_names[i], preds[i]) for i in top3_idx]
    # Sonuç metni
    result = f"<div style='font-size:1.3em;'><b>Tahmin:</b> <span style='color:#2ecc71'>{predicted_class}</span> <b>({confidence*100:.2f}%)</b></div>"
    result += "<br><b>Top 3 Tahmin:</b><br>"
    result += """<table style='width:60%;margin:auto;background:#222;border-radius:8px;'><tr><th>Sıra</th><th>Tür</th><th>Güven</th></tr>"""
    for i, (cls, conf) in enumerate(top3, 1):
        result += f"<tr><td>{i}</td><td>{cls}</td><td>{conf*100:.2f}%</td></tr>"
    result += "</table>"
    return result, None

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """<h1 style='text-align:center; color:#e67e22;'>Oxford-IIIT Pet Sınıflandırıcı (Production Model)</h1>
        <div style='text-align:center; font-size:1.1em; color:#bbb;'>
        Bu uygulama, Oxford-IIIT Pet Dataset ile eğitilmiş MobileNetV2 tabanlı bir model kullanır.<br>
        Kullanım: Bir pet resmi yükleyin, model türünü ve güven skorunu tahmin etsin.<br>
        Top 3 tahmin ve güven skorları tablo olarak gösterilir.
        </div>
        """
    )
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Pet Resmi Yükleyin")
            submit_btn = gr.Button("Tahmin Et", elem_id="submit-btn", scale=2)
        with gr.Column():
            output = gr.HTML(label="Tahmin Sonucu")
    submit_btn.click(fn=predict, inputs=image_input, outputs=output)

if __name__ == "__main__":
    demo.launch() 