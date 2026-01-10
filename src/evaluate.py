import torch
import numpy as np
import argparse
import os
import sys
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append('src')

from data.voc_dataset import VOCDataset, collate_fn
from data.transforms import build_transforms
from model.detector import FCOSDetector

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle']

def calculate_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate(model, val_loader, device, save_results=True):
    model.eval()
    
    gt_db = {i: {} for i in range(5)} 
    det_db = {i: [] for i in range(5)}
    
    print("Loading GTs...")
    count_gt = 0
    
    print("Running Inference...")
    with torch.no_grad():
        img_idx = 0
        for images, boxes_list, labels_list in tqdm(val_loader):
            images = images.to(device)
            
            outputs = model(images)
            
            for i in range(len(images)):
                gt_boxes = boxes_list[i].numpy()
                gt_labels = labels_list[i].numpy()
                
                for gb, gl in zip(gt_boxes, gt_labels):
                    if img_idx not in gt_db[gl]:
                        gt_db[gl][img_idx] = []
                    gt_db[gl][img_idx].append({'bbox': gb, 'used': False})
                    count_gt += 1
                
                det_boxes = outputs[i]['boxes'].cpu().numpy()
                det_scores = outputs[i]['scores'].cpu().numpy()
                det_labels = outputs[i]['labels'].cpu().numpy()
                
                for db, ds, dl in zip(det_boxes, det_scores, det_labels):
                    det_db[dl].append({'bbox': db, 'score': ds, 'image_id': img_idx})
                
                img_idx += 1
    
    print("Calculating mAP...")
    aps = []
    per_class_ap = {}
    
    for c in range(5):
        dets = det_db[c]
        gts = gt_db[c]
        
        npos = sum([len(gts[img_id]) for img_id in gts])
        
        if npos == 0:
            aps.append(0.0)
            per_class_ap[CLASSES[c]] = 0.0
            print(f"Class {c} ({CLASSES[c]}): AP = 0.0000 (no GT)")
            continue
            
        dets = sorted(dets, key=lambda x: x['score'], reverse=True)
        
        TP = np.zeros(len(dets))
        FP = np.zeros(len(dets))
        
        for d in range(len(dets)):
            det = dets[d]
            image_id = det['image_id']
            bb = det['bbox']
            
            if image_id not in gts:
                FP[d] = 1.0
                continue
                
            gt_objs = gts[image_id]
            
            ovmax = -float('inf')
            jmax = -1
            
            bbgt = np.array([o['bbox'] for o in gt_objs])
            if bbgt.size > 0:
                ixmin = np.maximum(bbgt[:, 0], bb[0])
                iymin = np.maximum(bbgt[:, 1], bb[1])
                ixmax = np.minimum(bbgt[:, 2], bb[2])
                iymax = np.minimum(bbgt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (bbgt[:, 2] - bbgt[:, 0]) * (bbgt[:, 3] - bbgt[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            
            if ovmax > 0.5:
                if not gt_objs[jmax]['used']:
                    TP[d] = 1.
                    gt_objs[jmax]['used'] = True
                else:
                    FP[d] = 1.
            else:
                FP[d] = 1.
                
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = acc_TP / np.maximum(acc_TP + acc_FP, np.finfo(np.float64).eps)
        
        ap = calculate_ap(rec, prec)
        aps.append(ap)
        per_class_ap[CLASSES[c]] = float(ap)
        print(f"Class {c} ({CLASSES[c]}): AP = {ap:.4f}")
        
    mAP = np.mean(aps)
    print(f"Mean AP: {mAP:.4f}")
    
    if save_results:
        os.makedirs('results', exist_ok=True)
        results = {
            'mean_ap': float(mAP),
            'per_class_ap': per_class_ap
        }
        with open('results/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Results saved to results/evaluation_results.json")
    
    return mAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/VOC2012_train_val/VOC2012_train_val')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_dataset = VOCDataset(args.data_root, image_set='val', 
                             transform=build_transforms(is_train=False))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    model = FCOSDetector(num_classes=5, use_gn=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    
    evaluate(model, val_loader, device)
