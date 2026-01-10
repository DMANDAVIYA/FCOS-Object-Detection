import torch
import cv2
import numpy as np
import os
import sys
import time
import argparse
from pathlib import Path

sys.path.append('src')

from model.detector import FCOSDetector
from data.transforms import build_transforms

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']
COLORS = [
    (255, 107, 107),
    (78, 205, 196),
    (255, 230, 109),
    (255, 159, 243),
    (84, 160, 255)
]

def draw_detection(img, box, label, score, color):
    """Draw bounding box with label"""
    x1, y1, x2, y2 = box.astype(int)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    text = f"{label} {score:.2f}"
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    
    cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

def process_image(model, image_path, device, thresh=0.3, save_path=None):
    """Run inference on single image"""
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not read {image_path}")
        return None
        
    img_draw = img_orig.copy()
    h_orig, w_orig = img_orig.shape[:2]
    
    transform = build_transforms(is_train=False)
    img_tensor, _, _ = transform(img_orig)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        results = model(img_tensor)
    inference_time = time.time() - start_time
    
    res = results[0]
    boxes = res['boxes'].cpu().numpy()
    scores = res['scores'].cpu().numpy()
    labels = res['labels'].cpu().numpy()
    
    boxes[:, 0::2] *= (w_orig / 512.0)
    boxes[:, 1::2] *= (h_orig / 512.0)
    
    count = 0
    for i in range(len(boxes)):
        if scores[i] < thresh:
            continue
        draw_detection(img_draw, boxes[i], CLASSES[labels[i]], scores[i], COLORS[labels[i]])
        count += 1
    
    if save_path:
        cv2.imwrite(save_path, img_draw)
        print(f"Saved to {save_path}")
    
    return img_draw, count, inference_time

def batch_inference(checkpoint, image_dir, output_dir='results/detections', thresh=0.3, max_images=None):
    """Run inference on directory of images"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FCOSDetector(num_classes=5)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
    
    if max_images:
        image_files = image_files[:max_images]
    
    total_time = 0
    total_detections = 0
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in image_files:
        save_path = os.path.join(output_dir, f"det_{img_path.name}")
        _, count, inf_time = process_image(model, str(img_path), device, thresh, save_path)
        total_time += inf_time
        total_detections += count
    
    avg_fps = len(image_files) / total_time if total_time > 0 else 0
    
    print(f"\nInference Complete:")
    print(f"  Total Images: {len(image_files)}")
    print(f"  Total Detections: {total_detections}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Results saved to: {output_dir}")
    
    # Save FPS to results
    os.makedirs('results', exist_ok=True)
    import json
    eval_results_path = 'results/evaluation_results.json'
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    results['fps'] = float(avg_fps)
    results['inference_time_per_image'] = float(total_time / len(image_files)) if len(image_files) > 0 else 0
    
    with open(eval_results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return avg_fps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/detections')
    parser.add_argument('--thresh', type=float, default=0.3)
    parser.add_argument('--max_images', type=int, default=None, help='Limit number of images to process')
    
    args = parser.parse_args()
    batch_inference(args.checkpoint, args.image_dir, args.output_dir, args.thresh, args.max_images)
