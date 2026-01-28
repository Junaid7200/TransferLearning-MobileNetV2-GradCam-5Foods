# Notes

## Summary of the project

I built a 5‑class food classifier using transfer learning with MobileNetV2. I froze the pretrained ImageNet backbone and added a Global Average Pooling layer and a 5‑class softmax head. Because the dataset was very small, I used strong data augmentation and a deterministic train/val/test split to keep results reproducible. The model reached about 96% accuracy on a 27‑image test set, but I noted the test set is tiny so the result is indicative, not definitive. I also implemented Grad‑CAM to visualize which image regions drove predictions, which helps interpretability and catches spurious cues.

## Common Questions

- Transfer Learning VS FineTuning:
  - transfer learning is when you have a pretrained backbone and you freeze it and then add a new head simply. FineTuning is when you unfreeze some or all of the backbone and train it all on your dataset.
- Why use Transfer Learning here
  - FineTuning would have probably lead to overfitting because the dataset is very small so I had to take every precaution to make sure it doesn't overfit so transfer learning would be way better then finetuning.
- If you wanted the model to work on the original 1000 classes as well as the new 5, how would that be achieved?
  - I would need to include the original 1000 classes plus my 5 classes in the dense layer at the end. The current train_gen.num_classes outputs 5, it should output 1005.
- What’s a kernel?
  - In a convolution layer, a kernel is a small matrix of learnable weights (e.g., 3x3) that slides over the input image.
  - At each position, it computes a dot product between the kernel and the local patch to produce one value in the feature map.
  - Multiple kernels learn different patterns (edges, textures, shapes), producing multiple feature maps.

## Rapid-Fire questions

- What is transfer learning?
  - Using a pretrained model’s features and training a new head on your data.

- Transfer learning vs fine‑tuning?
  - Transfer = freeze base; fine‑tune = unfreeze some layers.

- Why not fine‑tune here?
  - Tiny dataset; freezing reduces overfitting.

- Can your model predict ImageNet + 5?
  - No, you replaced the head with 5‑class output.

- How to make 1005?
  - Use a 1005‑class head and train with data for all 1005.

- Why MobileNetV2?
  - Lightweight, fast, strong pretrained features.

- What’s a kernel?
  - Small learned filter that detects local patterns.

- CNN vs ANN?
  - CNN uses local receptive fields and weight sharing.

- Why GAP instead of Flatten?
  - Fewer parameters, less overfitting.

- Why preprocess_input?
  - Match the pretrained model’s expected pixel scaling.

- Why softmax + cross‑entropy?
  - Multi‑class classification with mutually exclusive labels.

- What does data augmentation do?
  - Increases effective diversity and reduces overfitting.

- Why deterministic split?
  - Reproducibility and fair evaluation.

- What’s Grad‑CAM for?
  - Interpretability—visualizing decision regions.

- Main limitation of your results?
  - Tiny test set, so accuracy is optimistic.
