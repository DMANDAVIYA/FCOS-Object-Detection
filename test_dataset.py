
import sys
import os
import torch
from torch.utils.data import DataLoader

# Add src to path so we can import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.voc_dataset import VOCDataset, collate_fn
from data.transforms import build_transforms

def test_dataset():
    # Root path as discovered
    root = r'd:/A5/interviews/sapien/custom_objdetect/data/VOC2012_train_val/VOC2012_train_val'
    
    print(f"Testing dataset at: {root}")
    
    # Init dataset
    ds = VOCDataset(root, image_set='train', transform=build_transforms(is_train=True), mosaic_prob=0.5)
    
    print(f"Dataset length: {len(ds)}")
    
    if len(ds) == 0:
        print("Error: Dataset is empty!")
        return

    # Get one item
    img, boxes, labels = ds[0]
    print(f"Item 0 shape: Img {img.shape}, Boxes {boxes.shape}, Labels {labels.shape}")
    
    # Test Loader
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    for i, (imgs, boxes, labels) in enumerate(loader):
        print(f"Batch {i} shape: Img {imgs.shape}")
        print(f"Batch {i} boxes types: {[b.shape for b in boxes]}")
        break  # Test just one batch

if __name__ == "__main__":
    test_dataset()
