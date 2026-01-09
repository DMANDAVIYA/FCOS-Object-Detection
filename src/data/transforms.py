import torch
import cv2
import numpy as np
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Resize(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        h, w, _ = img.shape
        scale = self.size / max(h, w)
        img = cv2.resize(img, (self.size, self.size)) # Square resize for simplicity in batching
        
        # But wait, plain resize distorts aspect ratio. 
        # Standard practice: Resize longest edge, pad short edge.
        # For simplicity in this assignment, let's just resize and adjust boxes.
        # (Mosaic output is already square, so this is mostly for non-mosaic path)
        
        # Re-calc scale factors
        scale_x = self.size / w
        scale_y = self.size / h
        
        if boxes is not None and len(boxes) > 0:
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y
            
        return img, boxes, labels

class ToTensor(object):
    def __call__(self, img, boxes=None, labels=None):
        # Convert IMG from (H, W, C) [0-255] to (C, H, W) [0.0-1.0]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, boxes, labels

class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
    def __call__(self, img, boxes=None, labels=None):
        img = (img - self.mean) / self.std
        return img, boxes, labels

def build_transforms(is_train=True, img_size=512):
    if is_train:
        return Compose([
            Resize(img_size),
            ToTensor(),
            Normalize()
        ])
    else:
        return Compose([
            Resize(img_size),
            ToTensor(),
            Normalize()
        ])
