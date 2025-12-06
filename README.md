# Food Classifier (MobileNetV2)
- Transfer-learning notebook that classifies 5 foods (`biryani`, `chapli_kebab`, `chocolate_cake`, `samosa`, `seekh_kebab`) using a frozen MobileNetV2 backbone and a lightweight GAP + softmax head.
- Data lives under `Data/` (augmented) and `Data_original/` (raw). Validation split is 25 images (5 classes).
- Core work is in the notebooks: `main.ipynb` (exploration/baselines) and `main2.ipynb` (cleaned-up run with the augmented set).

## Data
- Original: `Data_original/Train` (109 imgs) and `Data_original/Validation` (25 imgs) with balanced folders per class.
- Augmented: `Data/Train` contains the originals plus ~4 saved augmentations per image (~545 files). `Data/Validation` keeps the 25 clean validation images.
- Augmentations: rotations ±30°, horizontal flips, zoom 0.2, width/height shifts 0.15, brightness 0.8–1.2, shear 0.1, nearest fill.

## Modeling
- Backbone: `tensorflow.keras.applications.MobileNetV2` pretrained on ImageNet (`include_top=False`, input 224×224×3), kept frozen.
- Head: `GlobalAveragePooling2D` → `Dense(5, softmax)`.
- Training: Adam, categorical cross-entropy, batch sizes 32 (train) / 16 (val), 5 epochs.

## Experiments (see notebooks)
- `main.ipynb`: Baseline feature-extraction on the raw set, then on-the-fly augmentation. Training logs showed high val accuracy, but a shuffled validation generator led to poor classification reports (≈12–20% acc), revealing weak generalization/eval misalignment.
- `main2.ipynb`: Re-trained on the saved augmented set (`Data/Train`). Five epochs reached a perfect classification report on the 25-image validation split and predicts `cake.jpg` as `chocolate_cake` with 0.93 confidence. Given the tiny val set, treat the 100% score as optimistic and still prone to overfitting.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
# Install deps (uv lock is included):
pip install uv             # if you want the exact locked set
uv sync
# or with pip: pip install tensorflow matplotlib numpy pandas scikit-learn seaborn
jupyter notebook main2.ipynb
```

## Next ideas
- Freeze shuffling when scoring to align `y_true`/`y_pred`, add a held-out test set, and unfreeze top MobileNetV2 blocks for fine-tuning.
- Add Grad-CAM for interpretability and compare against another backbone (e.g., EfficientNet) on a larger or more fine-grained food dataset.
