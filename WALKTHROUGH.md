# FCOS Implementation Walkthrough

This document explains the complete architecture and training pipeline of the FCOS object detector.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Model Components](#model-components)
3. [Training Pipeline](#training-pipeline)
4. [Loss Functions](#loss-functions)
5. [Data Pipeline](#data-pipeline)
6. [Evaluation](#evaluation)

---

## Architecture Overview

FCOS is an anchor-free, per-pixel prediction object detector. Unlike traditional detectors (Faster R-CNN, YOLO), it doesn't rely on predefined anchor boxes.

**Key Idea**: For each pixel in the feature map, predict:
1. Classification score (5 classes)
2. Bounding box offsets (left, top, right, bottom distances)
3. Centerness score (quality measure)

**Pipeline Flow**:
```
Input Image (HxWx3)
    ↓
ResNet-18 Backbone (with Group Norm)
    ↓
C3, C4, C5 features
    ↓
Feature Pyramid Network (FPN)
    ↓
P3, P4, P5, P6, P7 (multi-scale features)
    ↓
FCOS Head (shared across all levels)
    ↓
Per-pixel predictions: [class, bbox, centerness]
    ↓
Post-processing (NMS)
    ↓
Final Detections
```

---

## Model Components

### 1. Backbone (`src/model/backbone.py`)

**ResNet-18 with Group Normalization**

```python
Input: (B, 3, H, W)
    ↓
Conv1 + GN + ReLU
    ↓
Layer1 (64 channels)
    ↓
Layer2 (128 channels) → C3 output
    ↓
Layer3 (256 channels) → C4 output
    ↓
Layer4 (512 channels) → C5 output
```

**Why Group Norm?**
- BatchNorm fails with small batch sizes (2-4)
- GN divides channels into groups, normalizes independently
- Stable training even with batch_size=1

**Output**: 
- C3: 128 channels, stride 8
- C4: 256 channels, stride 16
- C5: 512 channels, stride 32

### 2. Feature Pyramid Network (`src/model/fpn.py`)

**Purpose**: Create multi-scale features for detecting objects of different sizes

```python
C5 (512ch, 1/32) ──→ P5 (256ch, 1/32)
    ↑                     ↓
C4 (256ch, 1/16) ──→ P4 (256ch, 1/16)
    ↑                     ↓
C3 (128ch, 1/8)  ──→ P3 (256ch, 1/8)

P5 ──→ P6 (256ch, 1/64)  [MaxPool]
P6 ──→ P7 (256ch, 1/128) [MaxPool]
```

**Scale Assignment**:
- P3: Small objects (8-64 pixels)
- P4: Medium objects (64-128 pixels)
- P5: Large objects (128-256 pixels)
- P6, P7: Very large objects (256+ pixels)

### 3. FCOS Head (`src/model/head.py`)

**Shared Head Architecture**:
```
Input: P3/P4/P5/P6/P7 (256 channels each)
    ↓
Classification Tower:
    Conv(256) + GN + ReLU  ×4
    ↓
    Conv(num_classes)
    
Regression Tower:
    Conv(256) + GN + ReLU  ×4
    ↓
    Conv(4)  [bbox: l,t,r,b]
    Conv(1)  [centerness]
```

**Per-Level Scaling**:
Each FPN level has a learnable scale parameter to adjust bbox predictions.

**Output Shapes** (for 512x512 input):
- P3: (B, 5, 64, 64), (B, 4, 64, 64), (B, 1, 64, 64)
- P4: (B, 5, 32, 32), (B, 4, 32, 32), (B, 1, 32, 32)
- P5: (B, 5, 16, 16), (B, 4, 16, 16), (B, 1, 16, 16)
- P6: (B, 5, 8, 8), (B, 4, 8, 8), (B, 1, 8, 8)
- P7: (B, 5, 4, 4), (B, 4, 4, 4), (B, 1, 4, 4)

---

## Training Pipeline

### 1. Data Loading (`src/data/voc_dataset.py`)

**Augmentation Strategy**:
```python
if random() < 0.5:
    image = Mosaic(4 images)  # Stitch 4 images into one
    
if random() < 0.5:
    image = MixUp(image, another_image)  # Blend two images

image = Resize(512, 512)
image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Mosaic Augmentation**:
```
┌─────────┬─────────┐
│ Image 1 │ Image 2 │
├─────────┼─────────┤
│ Image 3 │ Image 4 │
└─────────┴─────────┘
```
- Increases small object detection
- Forces model to learn partial objects

**MixUp Augmentation**:
```
Output = α * Image1 + (1-α) * Image2
```
- Regularization technique
- Reduces overfitting

### 2. Training Loop (`src/train.py`)

**Per Epoch**:
```python
1. Training Phase:
   for batch in train_loader:
       - Forward pass
       - Calculate loss (cls + reg + cnt)
       - Backward pass
       - Gradient clipping (max_norm=1.0)
       - Optimizer step
       
2. Validation Phase:
   - Run full evaluation on val set
   - Calculate mAP@0.5
   
3. Logging:
   - Save checkpoint: fcos_epoch_N.pth
   - Update training_history.csv
   - If mAP improved: save best_model.pth
```

**Optimizer**: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Cosine annealing scheduler

---

## Loss Functions

### 1. Focal Loss (`src/model/loss.py`)

**Purpose**: Handle class imbalance (most pixels are background)

```python
FL(p) = -α(1-p)^γ * log(p)

α = 0.25  # Weight for positive class
γ = 2.0   # Focusing parameter
```

**Effect**: Down-weights easy examples, focuses on hard negatives

### 2. GIoU Loss

**Purpose**: Better bounding box regression than L1/L2

```python
IoU = Intersection / Union
GIoU = IoU - (C - Union) / C

where C = smallest enclosing box
```

**Advantages**:
- Handles non-overlapping boxes
- Provides gradient even when IoU=0
- Faster convergence

### 3. Centerness Loss

**Purpose**: Suppress low-quality detections far from object center

```python
centerness = sqrt((min(l,r) / max(l,r)) * (min(t,b) / max(t,b)))
```

**Loss**: Binary Cross Entropy

---

## Data Pipeline

### VOC Dataset Structure
```
data/VOC2012_train_val/VOC2012_train_val/
├── Annotations/
│   └── *.xml
├── JPEGImages/
│   └── *.jpg
└── ImageSets/Main/
    ├── train.txt
    └── val.txt
```

### Label Assignment

For each ground truth box at each FPN level:

```python
1. Check if pixel location is inside GT box
2. Calculate distances: left, top, right, bottom
3. Check if distances are within level's scale range:
   - P3: [0, 64]
   - P4: [64, 128]
   - P5: [128, 256]
   - P6: [256, 512]
   - P7: [512, ∞]
4. If multiple GTs match, assign to smallest area GT
```

---

## Evaluation

### mAP Calculation (`src/evaluate.py`)

**Process**:
```python
1. Run inference on all validation images
2. For each class:
   a. Sort predictions by confidence
   b. Match predictions to ground truth (IoU > 0.5)
   c. Calculate precision-recall curve
   d. Compute Average Precision (area under PR curve)
3. mAP = mean of all class APs
```

**Metrics Logged**:
- Per-class AP
- Mean AP (mAP)
- Number of ground truth instances per class

### Inference Post-Processing

```python
1. Decode bbox predictions:
   x1 = x - left
   y1 = y - top
   x2 = x + right
   y2 = y + bottom
   
2. Filter by score threshold (0.05)

3. Apply NMS (IoU threshold 0.6):
   - Sort by score
   - Suppress overlapping boxes
   
4. Keep top 100 detections
```

---

## Key Design Decisions

### 1. Why Group Normalization?
- Small batch sizes (2-4) due to memory constraints
- BatchNorm statistics unreliable with small batches
- GN provides stable training

### 2. Why 5 FPN Levels?
- Handles objects from 8px to 512px
- More levels = better multi-scale detection
- Computational cost is manageable

### 3. Why Mosaic + MixUp?
- VOC is a small dataset (~5000 train images)
- Strong augmentation prevents overfitting
- Mosaic helps with small object detection

### 4. Why Save Every Epoch?
- Training can be unstable
- Best model might not be the last epoch
- Allows post-training checkpoint selection

---

## Performance Expectations

**Training Metrics**:
- Initial loss: ~10-15
- Final loss: ~1.5-2.0
- Training time: 2-3 hours (50 epochs, GPU)

**Validation Metrics**:
- Expected mAP@0.5: 0.35-0.45
- Per-class AP variance: 0.2-0.5

**Common Issues**:
- Loss spike at epoch 5-10: Normal (learning rate adjustment)
- mAP plateau after epoch 30: Expected (model converged)
- High loss on small objects: Inherent difficulty

---

## References

- FCOS Paper: https://arxiv.org/abs/1904.01355
- Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/
- Group Normalization: https://arxiv.org/abs/1803.08494
