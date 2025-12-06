# Food Classifier (MobileNetV2)
- 5-class food classifier (`biryani`, `chapli_kebab`, `chocolate_cake`, `samosa`, `seekh_kebab`) using a frozen MobileNetV2 backbone with a GAP + softmax head.
- Deterministic data split and evaluation live in `main3.ipynb` (portfolio-ready). Earlier explorations remain in `main.ipynb` / `main2.ipynb`.
- Grad-CAM visualizations included for interpretability.

## Data
- Raw: `Data_original/Train` (109 imgs) + `Data_original/Validation` (25 imgs), balanced across 5 classes.
- Deterministic split (seed 42) built into `main3.ipynb`:
  - `Data_split/train`: 80 images (≈14/13/23/17/13 per class)
  - `Data_split/val`: 27 images (5/5/7/5/5)
  - `Data_split/test`: 27 images (5/5/7/5/5)
- Augmentation (train only): rotation ±30°, horizontal flip, zoom 0.2, width/height shift 0.15, brightness 0.8–1.2, shear 0.1, nearest fill.

## Modeling
- Backbone: `tensorflow.keras.applications.MobileNetV2` (`include_top=False`, ImageNet weights), frozen.
- Head: `GlobalAveragePooling2D` → `Dense(5, softmax)`.
- Training: Adam, categorical cross-entropy, batch size 32 (train) / 16 (val/test), 5 epochs, seeds fixed for reproducibility.

## Results (main3.ipynb, test split = 27 images)
- Test accuracy: **96.3%**
- Per-class (precision/recall): all perfect except `samosa` (0.83/1.00) and `seekh_kebab` (1.00/0.80).
- Confusion matrix and Grad-CAM overlays are rendered in-notebook.
- Note: test set is small; results are indicative, not definitive.

## How to run
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install deps (uv lock included)
pip install uv
uv sync
# or: pip install tensorflow matplotlib numpy pandas scikit-learn seaborn

jupyter notebook main3.ipynb
```
- To avoid re-writing the deterministic split, skip the split cell when re-running; otherwise it recreates `Data_split/`.
- Run cells in order: data inventory → split → generators (shuffle False for val/test) → seeds → model build/train → test eval → learning curves → Grad-CAM (regular and strong overlay).

## Interpretability
- Grad-CAM helper + “strong overlay” cell show original, overlay, and raw heatmap with predicted label/confidence.
- Uses cached grad model and higher alpha to make activations visible; colormap uses the Matplotlib non-deprecated API.

## Next steps (ideas)
- Fine-tune upper MobileNetV2 blocks; compare EfficientNet/ResNet backbones.
- Add a true hold-out test with more images; run multi-seed averages.
- Save/export artifacts (`SavedModel`, history JSON) and add a simple inference script/Gradio demo.
