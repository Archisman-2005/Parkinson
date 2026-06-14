# Parkinson's Disease Detection via Handwriting & Drawing Pattern Analysis

> **Summer Research Project** — Image & Signal Processing Lab, Dept. of Electrical Engineering, IIT Kharagpur
> \n**Model Link:** https://drive.google.com/file/d/1k567FnEYvRI5T9jBv2Oqr2nUvsWytWda/view?usp=drive_link 

---

## Overview

This project develops a **two-stage deep learning pipeline** for non-invasive Parkinson's Disease (PD) detection from handwriting and drawing samples. The system first identifies the type of drawing task, then performs binary Healthy Control (HC) vs. Parkinson's Disease (PD) classification — enabling a fully automated diagnostic aid without requiring clinical examination.

PD manifests in handwriting as measurable neuromotor degradation: reduced stroke smoothness, micrographia, spiral tremor, and loss of directional control. This project aims to quantify these biomarkers computationally across 8 standardised drawing tasks.

---

## Architecture

```
User Drawing Input
       │
       ▼
┌─────────────────────┐
│   Stage 1: CNN      │  EfficientNetB0 — 8-class pattern classifier
│   Pattern Router    │  (spiral / meander / wave / circle /
│                     │   ellipse / figure-eight / arabic / latin)
└─────────┬───────────┘
          │  routes to pattern-specific model
          ▼
┌─────────────────────┐
│   Stage 2: ViT      │  Vision Transformer ensemble
│   HC/PD Classifier  │  7 pattern-specific binary classifiers
│                     │  → HC confidence / PD confidence
└─────────┬───────────┘
          │
          ▼
    Final Diagnostic Output
    (HC % / PD % with per-pattern breakdown)
```

---

## Datasets

| Dataset | Source | Contents |
|---|---|---|
| HandPD | IEEE Dataport | Spiral, meander drawings |
| NewHandPD | IEEE Dataport | Extended drawing tasks |
| PARKD | Public | Multi-task handwriting samples |

**Unified dataset after preprocessing:** ~4,150 images → ~3,860 balanced images across 8 categories

| Category | Approx. Samples (post-augmentation) |
|---|---|
| Spiral | 780 |
| Meander | 636 |
| Wave | 612 |
| Circle | 396 |
| Arabic | 360 |
| Eight | 360 |
| Ellipse | 360 |
| Latin | 360 |

---

## Stage 1 — CNN Pattern Classifier

### Model
- **Architecture:** EfficientNetB0 (ImageNet pretrained) + custom classification head
- **Parameters:** 7.84M total, 1.71M trainable
- **Input:** 224 × 224 RGB images

### Training Strategy
- **Phase 1:** Frozen EfficientNetB0 backbone, train classification head only
- **Phase 2:** Unfreeze top 20 layers, fine-tune end-to-end at reduced learning rate
- **Novel epoch selector:** 5-method weighted voting ensemble (minimum validation loss, maximum validation accuracy, elbow-point detection, generalisation gap minimisation, loss-ratio stability) — eliminates manual tuning
- **Class weighting:** sklearn balanced class weights to handle imbalance (spiral 780 vs. figure-eight 300)
- **Data split:** Subject-level stratified 70/20/10 train/val/test to prevent leakage between augmented variants

### Results ✅
| Metric | Value |
|---|---|
| Test Accuracy | **78%** |
| Macro F1 | **0.74** |
| Classes exceeding 0.80 F1 | **6 of 8** |
| Weakest class (Arabic) | 0.31 F1 — targeted for remediation |

### Augmentation
Task-aware policies designed to **preserve clinically meaningful PD biomarkers**:
- Spiral / meander: Gaussian noise injection only (chirality and tremor pattern must be preserved)
- Minority classes (arabic, circle, eight, ellipse, latin, wave): rotation, shear, scaling, stroke-width variation, morphological dilation — 5–6× multiplier

---

## Stage 2 — Vision Transformer HC/PD Classifier

### Architecture
- **7 pattern-specific Transformer models** (circle and ellipse share one model)
- **Multi-scale patch tokenisation:** each 224×224 image decomposed into 3×3, 6×6, and 9×9 grids → 126 tokens total
- **Token embedding:** each patch passed through shared EfficientNetB0 feature extractor → 1,280-dim embedding → sequence of (126, 1,280)
- **Transformer encoder:** 2-layer multi-head self-attention (8 heads), learnable positional embeddings, 512-dim feed-forward blocks
- **Output:** independent HC and PD sigmoid scores with per-pattern learnable decision thresholds (optimised via 0.30–0.70 sweep)

### Class Imbalance Handling
Feature-space SMOTE applied in mean-pooled CNN embedding space (HC:PD ratios up to 1:2.3 for spiral), with synthetic samples replicated across the 126-token sequence.

### Current Status 🔄
- Architecture designed and implemented
- Feature precomputation pipeline complete — (N, 126, 1,280) tensors cached to `.npz`
- Linear probing on CNN features: **61.5% ± 8.5% cross-validated accuracy** — confirms HC/PD separability in feature space
- Transformer training under active debugging (optimisation on small per-pattern datasets)

---

## Planned Work

- [ ] Fix Transformer convergence — reduce architecture capacity for small per-pattern sample sizes
- [ ] Random Forest feature importance analysis — identify task-independent PD biomarkers across all 7 patterns
- [ ] Ablation studies — CNN+GRU and CNN+BiLSTM vs. Transformer on identical splits
- [ ] Multi-image inference pipeline — accept 4 user-uploaded images, route each through Stage 1, aggregate Stage 2 outputs via attention-weighted instance pooling
- [ ] Mobile app integration (BIOS — Biomarker Input and Output System) via Expo React Native + Firebase Firestore

---

## Tech Stack

| Component | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras, PyTorch |
| Classical ML | scikit-learn, imbalanced-learn (SMOTE) |
| Data | NumPy, pandas, Google Drive API |
| Training | Google Colab (GPU), `.npz` feature caching |
| Visualisation | Matplotlib, Seaborn |

---

## Repository Structure

```
parkinsons-drawing-analysis/
│
├── dataset/                  # Dataset audit and preprocessing scripts
├── augmentation/             # Task-aware augmentation pipelines
├── stage1_cnn/               # EfficientNetB0 training notebooks
├── stage2_vit/               # Vision Transformer notebooks per pattern
├── features/                 # Precomputed .npz feature tensors (gitignored)
├── results/                  # Confusion matrices, classification reports
└── README.md
```

---

## Citation / Datasets

- HandPD / NewHandPD: [IEEE Dataport](https://ieee-dataport.org)
- PARKD: Available via public repository

---

## Author

**Archisman Ghosh**
B.Tech Electrical Engineering, IIEST Shibpur
Summer Research Intern, ISP Lab, IIT Kharagpur (May 2026 – Present)
[LinkedIn](https://www.linkedin.com/in/archisman-ghosh-438188318) · [GitHub](https://github.com/archisnoob)
