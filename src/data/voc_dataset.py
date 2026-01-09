import torch
import torch.utils.data as data
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
import random

# PASCAL VOC Class Map (subset of 5 classes as requested)
VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle'
)
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

class VOCDataset(data.Dataset):
    """
    Advanced VOC Dataset with Mosaic, MixUp, and classic augmentations.
    Designed for training from scratch on small data.
    """
    def __init__(self, root: str, image_set: str = 'train', transform=None, 
                 mosaic_prob: float = 0.5, mixup_prob: float = 0.5, img_size: int = 512):
        """
        Args:
            root: Path to the directory containing 'Annotations', 'JPEGImages', 'ImageSets'
            image_set: 'train', 'val', or 'trainval'
        """
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.img_size = img_size
        
        # Adjust paths based on standard VOC layout inside the root
        self._annopath = os.path.join(root, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(root, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(root, 'ImageSets', 'Main', '%s.txt')
        
        self.ids = list()
        self._load_ids()
        
    def _load_ids(self):
        # We need to map 'train' -> 'train' or 'trainval' file usually located in ImageSets/Main/
        # User has VOC2012_train_val/VOC2012_train_val/ImageSets/Main/
        split_file = self._imgsetpath % self.image_set
        
        if not os.path.exists(split_file):
            # Fallback or error if 'trainval' isn't exact name
             print(f"Warning: split file {split_file} not found. Trying 'trainval' if 'train' was requested.")
             if self.image_set == 'train':
                 split_file = self._imgsetpath % 'trainval'

        with open(split_file, 'r') as f:
            lines = f.readlines()
            
        # Pre-filter loop disabled for speed. Trusting standard split file.
        # for line in lines:
        #     img_id = line.strip()
        #     if self._has_valid_objects(img_id):
        #         self.ids.append(img_id)
        
        # Load all IDs directly
        self.ids = [line.strip() for line in lines]
        print(f"Loaded {len(self.ids)} images for split {self.image_set}")

    def _has_valid_objects(self, img_id: str) -> bool:
        ann_path = self._annopath % img_id
        if not os.path.exists(ann_path): return False
        try:
            target = ET.parse(ann_path).getroot()
            for obj in target.iter('object'):
                name = obj.find('name').text.lower().strip()
                if name in VOC_CLASSES:
                    return True
        except:
            return False
        return False

    def __getitem__(self, index: int):
        # 1. Apply Mosaic with probability (Training only)
        if self.image_set == 'train' and random.random() < self.mosaic_prob:
             img, boxes, labels = self._load_mosaic(index)
        else:
             img, boxes, labels = self._load_image_and_boxes(index)
        
        # 2. Apply MixUp with probability (Training only)
        if self.image_set == 'train' and random.random() < self.mixup_prob:
             idx2 = random.randint(0, len(self.ids) - 1)
             img2, boxes2, labels2 = self._load_image_and_boxes(idx2)
             # Resize img2 to match img current size if needed
             if img2.shape != img.shape:
                 img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
             img, boxes, labels = self._mixup(img, boxes, labels, img2, boxes2, labels2)

        # 3. Standard Transforms (Resize, ColorJitter, Normalize, ToTensor)
        if self.transform:
            # Transform expects (img, boxes, labels) and returns (img, boxes, labels)
            # or just img. Here we assume our custom transform handles all.
            # Convert to RGB (OpenCV is BGR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Handled in load
            img, boxes, labels = self.transform(img, boxes, labels)

        return img, boxes, labels

    def _load_image_and_boxes(self, index: int):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        
        img = cv2.imread(self._imgpath % img_id)
        if img is None:
            raise FileNotFoundError(f"Image not found: {self._imgpath % img_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult: continue
            name = obj.find('name').text.lower().strip()
            if name not in VOC_CLASSES: continue
            
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = [int(bbox.find(pt).text) - 1 for pt in pts]
            
            boxes.append(bndbox)
            labels.append(CLASS_TO_IDX[name])
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # If no boxes found (shouldn't happen due to filter, but safety)
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            
        return img, boxes, labels

    def _load_mosaic(self, index: int):
        # 4-tile mosaic
        labels4, boxes4 = [], []
        indices = [index] + [random.randint(0, len(self.ids) - 1) for _ in range(3)]
        
        s = self.img_size
        xc, yc = int(random.uniform(s * 0.5, s * 1.5)), int(random.uniform(s * 0.5, s * 1.5))
        
        # Canvas size: 2x image size (to fit 4 images)
        result_img = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8) 
        
        for i, idx in enumerate(indices):
            img, boxes, labels = self._load_image_and_boxes(idx)
            h, w, _ = img.shape
            
            # Place image logic
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
            elif i == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(yc + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            
            result_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            pad_w, pad_h = x1a - x1b, y1a - y1b
            
            if len(boxes) > 0:
                boxes_copy = boxes.copy()
                boxes_copy[:, 0] += pad_w
                boxes_copy[:, 2] += pad_w
                boxes_copy[:, 1] += pad_h
                boxes_copy[:, 3] += pad_h
                labels4.append(labels)
                boxes4.append(boxes_copy)

        if len(boxes4):
            boxes4 = np.concatenate(boxes4, 0)
            labels4 = np.concatenate(labels4, 0)
            np.clip(boxes4[:, 0::2], 0, 2 * s, out=boxes4[:, 0::2])
            np.clip(boxes4[:, 1::2], 0, 2 * s, out=boxes4[:, 1::2])
        else:
             boxes4 = np.zeros((0, 4), dtype=np.float32)
             labels4 = np.zeros((0,), dtype=np.int64)
        
        # Resize to target size (Mosaic produces 2x size, we shrink back)
        result_img = cv2.resize(result_img, (s, s))
        if len(boxes4) > 0:
            boxes4 /= 2.0
            
        return result_img, boxes4, labels4

    def _mixup(self, img1, boxes1, labels1, img2, boxes2, labels2):
        r = np.random.beta(32.0, 32.0)  # MixUp ratio
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        boxes = np.vstack((boxes1, boxes2))
        labels = np.hstack((labels1, labels2))
        return img, boxes, labels

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        boxes.append(torch.tensor(box) if isinstance(box, np.ndarray) else box)
        labels.append(torch.tensor(label) if isinstance(label, np.ndarray) else label)
    images = torch.stack(images, 0)
    return images, boxes, labels
