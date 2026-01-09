
import torch
import torch.nn as nn
from .backbone import ResNet18
from .fpn import FPN
from .head import FCOSHead
from .loss import FCOSLoss

class FCOSDetector(nn.Module):
    def __init__(self, num_classes=5, use_gn=True):
        super(FCOSDetector, self).__init__()
        
        # 1. Backbone
        self.backbone = ResNet18(use_gn=use_gn)
        # ResNet18 returns C3 (128), C4 (256), C5 (512)
        
        # 2. FPN
        self.fpn = FPN([128, 256, 512], out_channels=256)
        
        # 3. Head
        self.head = FCOSHead(in_channels=256, num_classes=num_classes, use_gn=use_gn)
        
        # 4. Loss
        self.loss_func = FCOSLoss()
        
    def forward(self, images, targets=None):
        """
        Args:
            images: [N, 3, H, W]
            targets: list of dicts (during training)
        """
        # Backbone
        c3, c4, c5 = self.backbone(images)
        
        # FPN
        features = self.fpn([c3, c4, c5]) # [P3, P4, P5, P6, P7]
        
        # Head
        logits, bbox_reg, centerness = self.head(features)
        
        if self.training and targets is not None:
            return self.loss_func(logits, bbox_reg, centerness, targets)
        else:
            # Inference Logic (Simple post-processing)
            return self._post_process(logits, bbox_reg, centerness, images.shape[-2:])

    def _post_process(self, logits, bbox_reg, centerness, img_size):
        # Decode and NMS
        # logits: list of [N, C, H, W]
        # bbox_reg: list of [N, 4, H, W]
        # centerness: list of [N, 1, H, W]
        
        detections = []
        strides = [8, 16, 32, 64, 128]
        
        # Collect all candidates from all levels
        all_boxes = []
        all_scores = []
        all_labels = []
        
        N = logits[0].shape[0]
        
        for l, stride in enumerate(strides):
            cls_score = logits[l].sigmoid()
            center_score = centerness[l].sigmoid()
            bbox_pred = bbox_reg[l]
            
            n, c, h, w = cls_score.shape
            
            # Grid
            shift_x = torch.arange(0, w, device=cls_score.device) * stride
            shift_y = torch.arange(0, h, device=cls_score.device) * stride
            y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            x = x + stride / 2
            y = y + stride / 2
            points = torch.stack((x, y), -1).reshape(-1, 2) # [H*W, 2]
            
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(n, -1, c)
            center_score = center_score.permute(0, 2, 3, 1).reshape(n, -1, 1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(n, -1, 4)
            
            # Apply centerness
            final_scores = torch.sqrt(cls_score * center_score)
            
            # Filter low confidence
            thresh = 0.05
            for i in range(n):
                scores_i = final_scores[i] # [HW, C]
                boxes_i = bbox_pred[i] # [HW, 4]
                points_i = points # [HW, 2]
                
                # Keep top K for speed
                # Flatten scores
                scores_flat = scores_i.flatten()
                
                # Filter indices
                keep_idxs = scores_flat > thresh
                if keep_idxs.sum() == 0:
                    continue
                    
                scores_keep = scores_flat[keep_idxs]
                # indices to (anchor_idx, class_idx)
                # anchor_idx = idx // C
                # class_idx = idx % C
                # But we flattened. 
                
                # Simpler approach: Max per row
                vals, idxs = scores_i.max(dim=1)
                mask = vals > thresh
                
                if mask.sum() == 0:
                    continue
                
                scores_per = vals[mask]
                class_per = idxs[mask]
                boxes_per = boxes_i[mask]
                points_per = points_i[mask]
                
                # Decode boxes
                l_ = boxes_per[:, 0]
                t_ = boxes_per[:, 1]
                r_ = boxes_per[:, 2]
                b_ = boxes_per[:, 3]
                
                x1 = points_per[:, 0] - l_
                y1 = points_per[:, 1] - t_
                x2 = points_per[:, 0] + r_
                y2 = points_per[:, 1] + b_
                
                boxes_decoded = torch.stack([x1, y1, x2, y2], dim=-1)
                
                # Clip to image
                boxes_decoded[:, 0::2].clamp_(0, img_size[1])
                boxes_decoded[:, 1::2].clamp_(0, img_size[0])
                
                if len(all_boxes) <= i:
                     all_boxes.append([])
                     all_scores.append([])
                     all_labels.append([])
                     
                while len(all_boxes) <= i: # Fix index error logic
                     all_boxes.append([])
                     all_scores.append([])
                     all_labels.append([])

                all_boxes[i].append(boxes_decoded)
                all_scores[i].append(scores_per)
                all_labels[i].append(class_per)

        # NMS
        results = []
        import torchvision
        for i in range(N):
            if i >= len(all_boxes) or len(all_boxes[i]) == 0:
                results.append({'boxes': torch.tensor([]), 'scores': torch.tensor([]), 'labels': torch.tensor([])})
                continue
            
            boxes = torch.cat(all_boxes[i], dim=0)
            scores = torch.cat(all_scores[i], dim=0)
            labels = torch.cat(all_labels[i], dim=0)
            
            # Batched NMS
            keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold=0.6)
            
            # Limit detections
            keep = keep[:100]
            
            results.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep]
            })
            
        return results
