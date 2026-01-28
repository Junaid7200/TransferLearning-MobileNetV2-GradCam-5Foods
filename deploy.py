from fastapi import FastAPI, UploadFile, File, Query
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
import json
from pathlib import Path

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
IMAGENET_LABELS_PATH = "imagenet_class_index.json"

# Load once at startup
model = tf.keras.models.load_model(MODEL_PATH)

# If multi-head, index 1 is food head
def predict_food_probs(batch):
    outputs = model([batch], training=False)
    if isinstance(outputs, (list, tuple)):
        return outputs[1].numpy()
    return outputs.numpy()

def predict_both_heads(batch):
    outputs = model([batch], training=False)
    if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
        imagenet_probs = outputs[0].numpy()[0]
        food_probs = outputs[1].numpy()[0]
        return imagenet_probs, food_probs
    # Fallback: single-head model
    single_probs = outputs.numpy()[0]
    return None, single_probs

def top_k_list(probs, labels, k):
    k = max(1, min(int(k), len(probs)))
    idxs = np.argsort(probs)[-k:][::-1]
    return [{"label": labels[i], "confidence": float(probs[i])} for i in idxs]

def load_imagenet_labels():
    path = Path(IMAGENET_LABELS_PATH)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # Expected format:
    # - {"0": ["n01440764", "tench"], ...} OR
    # - {"0": "tench, Tinca tinca", ...}
    labels = [None] * len(data)
    for k, v in data.items():
        idx = int(k)
        if isinstance(v, list) and len(v) > 1:
            label = v[1]
        else:
            label = str(v)
        labels[idx] = label
    return labels

IMAGENET_LABELS = load_imagenet_labels()

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
async def predict(
    file: UploadFile = File(...),
    food_top_k: int = Query(3, ge=1, le=5),
    imagenet_top_k: int = Query(5, ge=1, le=20),
):
    start = time.time()
    img_bytes = await file.read()
    batch = preprocess_image(img_bytes)
    imagenet_probs, food_probs = predict_both_heads(batch)

    # Food head (always returned)
    food_top = top_k_list(food_probs, CLASS_NAMES, food_top_k)
    food_top_label = food_top[0]["label"]
    food_top_conf = food_top[0]["confidence"]

    response = {
        "food": {
            "label": food_top_label,
            "confidence": food_top_conf,
            "top_k": food_top,
            "all_probs": {CLASS_NAMES[i]: float(p) for i, p in enumerate(food_probs)},
        },
        "latency_ms": round((time.time() - start) * 1000, 2),
    }

    # ImageNet head (optional if multi-head model)
    if imagenet_probs is not None:
        if IMAGENET_LABELS is None:
            imagenet_labels = [str(i) for i in range(len(imagenet_probs))]
        else:
            imagenet_labels = IMAGENET_LABELS
        response["imagenet"] = {
            "top_k": top_k_list(imagenet_probs, imagenet_labels, imagenet_top_k),
        }

    return response
