# Waste Material Segregation using CNNs

A convolutional neural network–based image classifier that segregates waste images into 7 categories to support automated recycling and sustainable waste management.

## Problem Statement

Manual waste sorting is labour-intensive, error-prone, and costly. Improper disposal contributes to environmental degradation. This project trains CNN models to classify waste images into the following categories:

1. Cardboard
2. Food Waste
3. Glass
4. Metal
5. Other
6. Paper
7. Plastic

## Dataset

- **Source**: `Dataset_Waste_Segregation/` (provided as a zipped folder)
- **Size**: 7,625 images
- **Image dimensions**: 256×256×3 (RGB), uniform across all samples
- **Class distribution**: Imbalanced (Plastic: 30.1%, Cardboard: 7.1%) — 4.25× ratio

## Approach

| Step | Choice | Rationale |
|------|--------|-----------|
| Image size | Resized to 128×128 | Faster training without losing class-distinguishing features |
| Normalisation | Pixel values scaled to [0, 1] | Standard CNN input pre-processing |
| Label encoding | One-hot (7 classes) | Required for multi-class softmax + categorical cross-entropy |
| Train/test split | 80/20 stratified | Preserves class proportions across splits |
| Class imbalance | `class_weight='balanced'` (sklearn) | Stops the model defaulting to the majority class (Plastic) |

## Model Configurations

Three CNN architectures, all with **3 convolutional blocks** (32, 64, 128 filters, 3×3 kernels, 'same' padding, ReLU + MaxPooling2D 2×2):

| Model | Regularization | Optimizer | Test Accuracy |
|-------|---------------|-----------|---------------|
| Baseline | L2(0.01) on Dense, Dropout(0.3) | SGD | 51.1% |
| Regularized | Conv Dropout(0.25) + L2(0.01) + Dropout(0.5), Dense(64) | SGD | 48.3% |
| **Augmented** | Dropout(0.3) only — augmentation as regularization | Adam | **53.2%** |

### Data Augmentation
- Rotation (±20°), width/height shift (±10%), horizontal flip, zoom (±15%)
- Pre-generated as numpy arrays (workaround for `ImageDataGenerator.flow()` incompatibility with Keras 3 / TF 2.21)
- Doubled the training set: 6,100 → 12,200 images

## Repository Contents

```
.
├── CNN_Assg_Waste_Segregation_Sai_Sreenivasan_PB.ipynb   # Main solution notebook
├── Dataset_Waste_Segregation-20260516T142040Z-3-001.zip  # Source dataset (zipped)
└── README.md                                              # This file
```

## How to Run

### Prerequisites
- Python 3.10+
- TensorFlow 2.18+
- NumPy, Matplotlib, Seaborn, scikit-learn, Pillow

### Steps
1. Clone the repository
2. Ensure the dataset zip file is at the project root
3. Open `CNN_Assg_Waste_Segregation_Sai_Sreenivasan_PB.ipynb` in Jupyter / VS Code
4. Run all cells sequentially

The notebook handles dataset extraction, training, evaluation, and visualisation end to end.

## Key Insights

- **Class weights significantly improve fairness**: Without them, Plastic recall reached 81% (the model essentially defaulted to the majority class). With class weights, predictions distributed more evenly across all 7 classes.
- **Regularization is a balance**: Heavy regularization (Model 2) eliminated overfitting but caused underfitting. Augmentation acts as implicit regularization, so combining it with explicit regularization (L2, conv dropout) hurt performance.
- **Optimizer choice depends on dataset size**: SGD generalises well on large datasets (CIFAR-10 50k images) but converges slowly on small ones. Adam handled the augmented small dataset better.
- **Hardest classes**: "Other" (no consistent visual pattern), "Paper" (similar to Cardboard), "Glass" (transparent — background dominates).

## Author

**Sai Sreenivasan PB**

Submitted as part of the Neural Networks CNN Assignment.
