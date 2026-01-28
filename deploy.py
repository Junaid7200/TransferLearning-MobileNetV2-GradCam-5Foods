from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time

app = FastAPI()

MODEL_PATH = "saved_model_dir/food_classifier.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "biryani",
    "chapli_kebab",
    "chocolate_cake",
    "samosa",
    "seekh_kebab",
]

# Load once at startup
model = tf.keras.models.load_model(MODEL_PATH)

# If multi-head, index 1 is food head
def predict_food_probs(batch):
    outputs = model([batch], training=False)
    if isinstance(outputs, (list, tuple)):
        return outputs[1].numpy()
    return outputs.numpy()

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocess: scale to [-1, 1]
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    img_bytes = await file.read()
    batch = preprocess_image(img_bytes)
    probs = predict_food_probs(batch)[0]
    top_idx = int(np.argmax(probs))
    top_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)
    top_conf = float(probs[top_idx])

    ranked = sorted(
        [
            {"label": CLASS_NAMES[i], "confidence": float(p)}
            for i, p in enumerate(probs)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "label": top_label,
        "confidence": top_conf,
        "top_k": ranked,
        "all_probs": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "latency_ms": round((time.time() - start) * 1000, 2),
    }
