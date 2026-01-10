# FCOS Object Detection: From-Scratch Implementation

Complete implementation of FCOS (Fully Convolutional One-Stage Object Detection) trained from random initialization on Pascal VOC 2012 dataset.

## Overview

This project demonstrates anchor-free object detection using FCOS architecture with:
- **ResNet-18 backbone** with Group Normalization
- **Feature Pyramid Network** (FPN) for multi-scale detection
- **Advanced data augmentation** (Mosaic + MixUp)
- **Specialized loss functions** (Focal Loss, GIoU Loss, Centerness)
- **From-scratch training** without pre-trained weights

**Performance:** mAP@0.5 = 0.1552 | FPS = 2.07 (CPU) | Parameters = 19.1M

## Quick Start

### 1. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Preparation
```bash
python setup_data.py
```
Downloads Pascal VOC 2012 dataset (5 classes: aeroplane, bicycle, bird, boat, bottle).

### 3. Training
```bash
python src/train.py
```
Trains for 50 epochs with per-epoch evaluation. Best model saved to `checkpoints/best_model.pth`.

### 4. Evaluation
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth
```
Calculates mAP and per-class AP. Results saved to `results/evaluation_results.json`.

### 5. Visualization
```bash
python visualize_training.py
```
Generates training curves (`results/training_curves.png`).

### 6. Inference
```bash
python batch_inference.py --max_images 5
```
Runs inference on validation images, measures FPS, saves visualizations to `results/detections/`.

### 7. Technical Report
```bash
cd d:/A5/interviews/sapien/custom_objdetect
pdflatex report_part1_intro.tex
pdflatex report_part2_architecture.tex
pdflatex report_part3_results.tex
```
The comprehensive 30-page technical report is split into 3 modular parts for easy compilation.

## Results Summary

| Metric | Value |
|--------|-------|
| **mAP@0.5** | 0.1552 |
| **Best Epoch** | 46 |
| **Model Size** | 72.92 MB |
| **Parameters** | 19,104,847 |
| **Inference FPS** | 2.07 (CPU) |

### Per-Class Performance
| Class | AP@0.5 |
|-------|--------|
| Aeroplane | 0.4812 |
| Bicycle | 0.1977 |
| Bird | 0.0470 |
| Boat | 0.0432 |
| Bottle | 0.0067 |

## Project Structure

```
custom_objdetect/
├── src/
│   ├── model/
│   │   ├── backbone.py          # ResNet-18 with Group Normalization
│   │   ├── fpn.py               # Feature Pyramid Network
│   │   ├── head.py              # FCOS detection head
│   │   └── detector.py          # Complete FCOS model
│   ├── data/
│   │   ├── voc_dataset.py       # Dataset loader
│   │   └── transforms.py        # Mosaic + MixUp augmentation
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── checkpoints/
│   ├── best_model.pth           # Best model checkpoint
│   └── training_history.csv     # Per-epoch metrics
├── results/
│   ├── training_curves.png      # Training visualization
│   ├── evaluation_results.json  # mAP and per-class AP
│   ├── training_summary.json    # Training statistics
│   └── detections/              # Inference visualizations
├── report_part1_intro.tex       # Technical report Part 1
├── report_part2_architecture.tex # Technical report Part 2
├── report_part3_results.tex     # Technical report Part 3
├── setup_data.py                # Data download script
├── visualize_training.py        # Training visualization
├── batch_inference.py           # Batch inference + FPS
└── requirements.txt             # Dependencies
```

## Architecture Details

### Backbone: ResNet-18 + Group Normalization
- Modified ResNet-18 with all BatchNorm replaced by GroupNorm (G=32)
- Enables stable training with small batch sizes
- Outputs multi-scale features: C3 (128ch), C4 (256ch), C5 (512ch)

### FPN: Multi-Scale Feature Pyramid
- 5 pyramid levels (P3-P7) with 256 channels each
- Handles objects from 8 to 512+ pixels
- Top-down pathway with lateral connections

### Detection Head
- Shared across all FPN levels
- Two parallel branches:
  - **Classification:** 4 conv layers → C classes (5)
  - **Regression:** 4 conv layers → bbox (4) + centerness (1)
- Learnable scale parameters per FPN level

### Loss Functions
1. **Focal Loss** (α=0.25, γ=2.0) - Handles class imbalance
2. **GIoU Loss** - Bounding box regression with gradient for non-overlapping boxes
3. **Centerness Loss** - Suppresses low-quality predictions

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| LR Schedule | Cosine Annealing |
| Batch Size | 16 |
| Epochs | 50 |
| Gradient Clipping | 1.0 |
| Augmentation | Mosaic (p=0.5) + MixUp (p=0.5) |

## Key Features

- **Anchor-Free Design:** No anchor boxes, simpler architecture
- **Group Normalization:** Batch-size independent normalization
- **Advanced Augmentation:** Mosaic and MixUp for data efficiency
- **Per-Epoch Evaluation:** Track mAP during training
- **Modular Implementation:** Clean, well-documented code
- **Comprehensive Documentation:** 30-page technical report with mathematical derivations

## Technical Report

The technical report is split into 3 parts for modularity:

1. **Part 1 (Introduction & Theory):** Background, related work, theoretical foundations
2. **Part 2 (Architecture & Loss):** Detailed architecture, loss function derivations
3. **Part 3 (Results & Analysis):** Experimental results, discussion, future work

Each part can be compiled independently or combined for the full 30-page report.

## Performance Analysis

### Why is mAP Low (0.1552)?

1. **Training from Scratch:** No ImageNet pre-training (expected gap: ~0.25 mAP)
2. **Limited Capacity:** ResNet-18 vs ResNet-50/101
3. **Small Dataset:** 5,717 images vs 1.2M for ImageNet
4. **Short Training:** 50 epochs vs 100-200 typical for from-scratch

### Comparison with Pre-trained Models
- This model (from scratch): **0.1552**
- FCOS + ResNet-50 (pre-trained): **~0.38-0.42**
- FCOS + ResNet-101 (pre-trained): **~0.42-0.45**

## Future Improvements

1. **Pre-training:** Use ImageNet weights (+0.20-0.25 mAP)
2. **Larger Backbone:** ResNet-50/101 (+0.10-0.15 mAP)
3. **Extended Training:** 100-200 epochs (+0.05-0.10 mAP)
4. **GPU Optimization:** TensorRT quantization (5-10x speedup)

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU training)
- See `requirements.txt` for full dependencies

## Citation

If you use this implementation, please cite the original FCOS paper:

```bibtex
@inproceedings{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={ICCV},
  year={2019}
}
```


