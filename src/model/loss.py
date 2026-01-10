
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ious(boxes1, boxes2):
    """
    Computes pairwise IoU matrix.
    Args:
        boxes1: [M, 4] (x1, y1, x2, y2)
        boxes2: [N, 4] (x1, y1, x2, y2)
    Returns:
        iou: [M, N]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

def giou_loss(pred_boxes, target_boxes, reduction='mean'):
    """
    GIoU Loss.
    """
    x1, y1, x2, y2 = pred_boxes.chunk(4, dim=-1)
    x1g, y1g, x2g, y2g = target_boxes.chunk(4, dim=-1)
    
    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)
    
    pred_area = (x2 - x1) * (y2 - y1)
    target_area = (x2g - x1g) * (y2g - y1g)
    
    x1_i = torch.max(x1, x1g)
    y1_i = torch.max(y1, y1g)
    x2_i = torch.min(x2, x2g)
    y2_i = torch.min(y2, y2g)
    
    inter_area = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    # Enclosing box
    x1_c = torch.min(x1, x1g)
    y1_c = torch.min(y1, y1g)
    x2_c = torch.max(x2, x2g)
    y2_c = torch.max(y2, y2g)
    area_c = (x2_c - x1_c).clamp(min=0) * (y2_c - y1_c).clamp(min=0)
    
    giou = iou - ((area_c - union_area) / (area_c + 1e-6))
    loss = 1 - giou
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

class FCOSLoss(nn.Module):
    def __init__(self, strides=(8, 16, 32, 64, 128)):
        super().__init__()
        self.strides = strides
    
    def __call__(self, logits, bbox_reg, centerness, targets):
        """
        Args:
            logits: list of [N, C, H, W]
            bbox_reg: list of [N, 4, H, W]
            centerness: list of [N, 1, H, W]
            targets: dict containing 'boxes' and 'labels'
        """
        self.device = logits[0].device
        
        # 1. Flatten all predictions
        all_logits = []
        all_bbox = []
        all_center = []
        all_points = []
        
        for l, stride in enumerate(self.strides):
            feat_h, feat_w = logits[l].shape[-2:]
            
            shift_x = torch.arange(0, feat_w, device=self.device) * stride
            shift_y = torch.arange(0, feat_h, device=self.device) * stride
            y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            x = x + stride / 2
            y = y + stride / 2
            points = torch.stack((x, y), -1).reshape(-1, 2)
            all_points.append(points)
            
            all_logits.append(logits[l].permute(0, 2, 3, 1).reshape(logits[l].size(0), -1, logits[l].size(1)))
            all_bbox.append(bbox_reg[l].permute(0, 2, 3, 1).reshape(bbox_reg[l].size(0), -1, 4))
            all_center.append(centerness[l].permute(0, 2, 3, 1).reshape(centerness[l].size(0), -1, 1))

        all_logits = torch.cat(all_logits, dim=1)
        all_bbox = torch.cat(all_bbox, dim=1)
        all_center = torch.cat(all_center, dim=1)
        all_points = torch.cat(all_points, dim=0)

        
        target_cls_list = []
        target_reg_list = []
        target_center_list = []
        
        for i in range(len(targets['boxes'])):
             gt_boxes = targets['boxes'][i].to(self.device)
             gt_labels = targets['labels'][i].to(self.device)
             
             if len(gt_boxes) == 0:
                 target_cls_list.append(torch.zeros_like(all_logits[i]))
                 target_reg_list.append(torch.zeros_like(all_bbox[i]))
                 target_center_list.append(torch.zeros_like(all_center[i]))
                 continue

            
             points = all_points 
             l = points[:, 0, None] - gt_boxes[:, 0]
             t = points[:, 1, None] - gt_boxes[:, 1]
             r = gt_boxes[:, 2] - points[:, 0, None]
             b = gt_boxes[:, 3] - points[:, 1, None]
             reg_targets = torch.stack([l, t, r, b], dim=2) 

             min_dist = reg_targets.min(dim=2)[0]
             is_in_box = min_dist > 0
             
             
             areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
             areas = areas[None, :].repeat(len(all_points), 1) 
             
        
             areas[~is_in_box] = float('inf')
             min_area, min_area_mod = areas.min(dim=1)
             
             pos_mask = min_area < float('inf')
             pos_ind = min_area_mod[pos_mask]
             
             labels = torch.zeros(all_logits.shape[1], 1, device=self.device, dtype=torch.long)
         
             labels_pos = gt_labels[pos_ind]
             
         
             cls_target = torch.zeros((all_logits.shape[1], all_logits.shape[2]), device=self.device)
             
             if len(pos_ind) > 0:
                  indices = torch.nonzero(pos_mask).squeeze(1)
                  cls_target[indices, labels_pos] = 1.0

             
             reg_target_final = torch.zeros_like(all_bbox[i])
             if len(pos_ind) > 0:
                 reg_target_pos = reg_targets[pos_mask, pos_ind, :]
                 reg_target_final[pos_mask] = reg_target_pos
             
             
             center_target_final = torch.zeros_like(all_center[i])
             if len(pos_ind) > 0:
                 l_ = reg_target_pos[:, 0]
                 t_ = reg_target_pos[:, 1]
                 r_ = reg_target_pos[:, 2]
                 b_ = reg_target_pos[:, 3]
                 
                 centerness_score = torch.sqrt((torch.min(l_, r_) / torch.max(l_, r_)) * 
                                               (torch.min(t_, b_) / torch.max(t_, b_)))
                 center_target_final[pos_mask, 0] = centerness_score
             
             target_cls_list.append(cls_target)
             target_reg_list.append(reg_target_final)
             target_center_list.append(center_target_final)

        target_cls = torch.stack(target_cls_list)
        target_reg = torch.stack(target_reg_list)
        target_center = torch.stack(target_center_list)
        
        pos_mask_all = target_center.squeeze(-1) > 0 
        num_pos = pos_mask_all.sum().clamp(min=1.0)
        
        loss_cls = sigmoid_focal_loss(all_logits, target_cls, reduction='sum') / num_pos
        
        if pos_mask_all.sum() > 0:
            pred_pos = all_bbox[pos_mask_all]
            target_pos = target_reg[pos_mask_all]
            
            points_pos = all_points.repeat(len(targets['boxes']), 1)[pos_mask_all.view(-1)]
            
            x1_p = points_pos[:, 0] - pred_pos[:, 0]
            y1_p = points_pos[:, 1] - pred_pos[:, 1]
            x2_p = points_pos[:, 0] + pred_pos[:, 2]
            y2_p = points_pos[:, 1] + pred_pos[:, 3]
            pred_boxes = torch.stack([x1_p, y1_p, x2_p, y2_p], dim=-1)
            
            x1_t = points_pos[:, 0] - target_pos[:, 0]
            y1_t = points_pos[:, 1] - target_pos[:, 1]
            x2_t = points_pos[:, 0] + target_pos[:, 2]
            y2_t = points_pos[:, 1] + target_pos[:, 3]
            target_boxes = torch.stack([x1_t, y1_t, x2_t, y2_t], dim=-1)
            
            loss_reg = giou_loss(pred_boxes, target_boxes, reduction='sum') / num_pos
            
            loss_center = F.binary_cross_entropy_with_logits(all_center[pos_mask_all], target_center[pos_mask_all], reduction='sum') / num_pos
        else:
            loss_reg = torch.tensor(0.0).to(self.device).float()
            loss_center = torch.tensor(0.0).to(self.device).float()
            
        return loss_cls, loss_reg, loss_center
