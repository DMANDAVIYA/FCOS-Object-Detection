# FCOS Object Detection - Custom Implementation

A from-scratch implementation of FCOS (Fully Convolutional One-Stage Object Detection) trained on Pascal VOC 2012 subset.

## Architecture

**Backbone**: ResNet-18 with Group Normalization  
**Neck**: Feature Pyramid Network (FPN) - Outputs P3, P4, P5, P6, P7  
**Head**: FCOS Detection Head with shared towers for classification and regression  
**Loss**: Focal Loss (classification) + GIoU Loss (regression) + BCE (centerness)

## Project Structure

```
custom_objdetect/
├── src/
│   ├── model/
│   │   ├── backbone.py      # ResNet-18 with GN
│   │   ├── fpn.py           # Feature Pyramid Network
│   │   ├── head.py          # FCOS detection head
│   │   ├── detector.py      # Main FCOS model
│   │   └── loss.py          # Loss functions
│   ├── data/
│   │   ├── voc_dataset.py   # VOC dataset with Mosaic/MixUp
│   │   └── transforms.py    # Image preprocessing
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation (mAP calculation)
│   └── find_best.py         # Find best checkpoint
├── data/                    # Dataset (extracted from data.zip)
├── checkpoints/             # Training outputs
├── requirements.txt
└── zip_src.py              # Package src for deployment
```

## Setup

### 1. Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Data

**Option A: Automatic Download (Recommended)**

1. Update `setup_data.py` with your Google Drive link:
```python
DRIVE_LINK = "https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing"
```

2. Run the setup script:
```bash
python setup_data.py
```

This will download `data.zip` from Google Drive and extract it automatically.

**Option B: Manual Download**

1. Download `data.zip` from the provided Google Drive link
2. Place it in the project root
3. Extract:
```bash
# Windows
Expand-Archive -Path data.zip -DestinationPath .

# Linux/Mac
unzip data.zip
```

**Expected structure after extraction**:
```
data/VOC2012_train_val/VOC2012_train_val/
├── Annotations/
├── JPEGImages/
└── ImageSets/Main/
```

## Training

### Basic Training
```bash
python src/train.py --data_root data/VOC2012_train_val/VOC2012_train_val --epochs 50 --batch_size 16
```

### Arguments
- `--data_root`: Path to VOC dataset root
- `--batch_size`: Training batch size (default: 4)
- `--epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 1e-4)
- `--save_dir`: Checkpoint directory (default: checkpoints)

### What Happens During Training
1. **Every Epoch**:
   - Full training pass with Mosaic + MixUp augmentation
   - Validation evaluation (mAP@0.5)
   - Checkpoint saved: `checkpoints/fcos_epoch_N.pth`
   - Metrics logged to: `checkpoints/training_history.csv`

2. **Best Model Tracking**:
   - Automatically saves `checkpoints/best_model.pth` when mAP improves

### Training Outputs
```
checkpoints/
├── fcos_epoch_1.pth
├── fcos_epoch_2.pth
├── ...
├── fcos_epoch_50.pth
├── best_model.pth
└── training_history.csv
```

**training_history.csv** contains:
- epoch
- train_loss
- cls_loss, reg_loss, cnt_loss
- mAP

## Evaluation

### Full Validation Set
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pth --data_root data/VOC2012_train_val/VOC2012_train_val
```

### Output
```
Class         | AP       | Count
-----------------------------------
aeroplane     | 0.4523   | 285
bicycle       | 0.3891   | 337
bird          | 0.3654   | 459
boat          | 0.2987   | 263
bottle        | 0.2145   | 469
-----------------------------------
Mean AP       | 0.3440
```

## Finding Best Checkpoint

If you have multiple checkpoints and want to find the best one:

```bash
python src/find_best.py
```

This runs a quick evaluation (100 images) on all checkpoints in `final_model/` and reports the winner.

## Key Features

### Data Augmentation
- **Mosaic**: 4-image stitching (50% probability)
- **MixUp**: Image blending (50% probability)
- Standard: Resize to 512x512, normalization

### Training Stability
- **Group Normalization**: Works with small batch sizes
- **Gradient Clipping**: Max norm 1.0
- **AdamW Optimizer**: Weight decay 1e-4
- **Cosine Annealing**: Learning rate scheduling

### Classes (VOC Subset)
1. aeroplane
2. bicycle
3. bird
4. boat
5. bottle

## Performance Notes

**Expected mAP@0.5**: 0.35-0.45 after 50 epochs on this 5-class subset

**Training Time**:
- GPU (T4): ~3 hours for 50 epochs
- Local GPU (RTX 3060): ~2 hours for 50 epochs

## Deployment

To package the code for deployment:
```bash
python zip_src.py
```

This creates `src.zip` containing all model code.

## Troubleshooting

**Out of Memory**:
- Reduce `--batch_size` to 2 or 1
- Reduce `num_workers` in `src/train.py` (line 24-25)

**Slow Evaluation**:
- Evaluation runs on full 5,823 validation images
- Takes ~1 hour on CPU, ~10 minutes on GPU

**Import Errors**:
- Ensure you're running from project root
- Check `sys.path.append('src')` is present in scripts
