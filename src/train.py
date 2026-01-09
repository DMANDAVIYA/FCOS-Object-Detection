import os
import sys
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

sys.path.append('src')

from data.voc_dataset import VOCDataset, collate_fn
from data.transforms import build_transforms
from model.detector import FCOSDetector
from evaluate import evaluate

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training started on {device}...")
    
    train_dataset = VOCDataset(args.data_root, image_set='train', transform=build_transforms(is_train=True), mosaic_prob=0.5, mixup_prob=0.5)
    val_dataset = VOCDataset(args.data_root, image_set='val', transform=build_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    model = FCOSDetector(num_classes=5, use_gn=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs(args.save_dir, exist_ok=True)
    history = []
    best_map = -1.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, cls_total, reg_total, cnt_total = 0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for images, boxes, labels in pbar:
            images = images.to(device)
            targets = {'boxes': [b.to(device) for b in boxes], 'labels': [l.to(device) for l in labels]}
            
            l_cls, l_reg, l_cnt = model(images, targets)
            loss = l_cls + l_reg + l_cnt
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            cls_total += l_cls.item()
            reg_total += l_reg.item()
            cnt_total += l_cnt.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()
        
        print(f"Evaluating Epoch {epoch}...")
        mAP = evaluate(model, val_loader, device)
        
        epoch_stats = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'cls_loss': cls_total / len(train_loader),
            'reg_loss': reg_total / len(train_loader),
            'cnt_loss': cnt_total / len(train_loader),
            'mAP': mAP
        }
        history.append(epoch_stats)
        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, 'training_history.csv'), index=False)
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'fcos_epoch_{epoch}.pth'))
        
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"NEW BEST mAP: {best_map:.4f} - Saved best_model.pth")

    print(f"Training complete. Best mAP: {best_map:.4f}. Logs in {args.save_dir}/training_history.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=r'd:\A5\interviews\sapien\custom_objdetect\data\VOC2012_train_val\VOC2012_train_val')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    train(args)
