import torch
import torch.nn as nn
import math

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return x * self.scale

class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_convs=4, use_gn=True):
        """
        Args:
            in_channels: FPN out channels
            num_classes: e.g. 5 for our subset
            num_convs: depth of the head towers
        """
        super(FCOSHead, self).__init__()
        
        cls_tower = []
        reg_tower = []
        
        for i in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels) if use_gn else nn.BatchNorm2d(in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
            
            reg_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            reg_tower.append(nn.GroupNorm(32, in_channels) if use_gn else nn.BatchNorm2d(in_channels))
            reg_tower.append(nn.ReLU(inplace=True))

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('reg_tower', nn.Sequential(*reg_tower))
        
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        for modules in [self.cls_tower, self.reg_tower, self.cls_logits, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        """
        Args:
           x: list of FPN features [P3, P4, P5, P6, P7]
        """
        logits = []
        bbox_reg = []
        centerness = []
        
        for l, feature in enumerate(x):
            cls_tower_out = self.cls_tower(feature)
            reg_tower_out = self.reg_tower(feature)
            
            logits.append(self.cls_logits(cls_tower_out))
            
            centerness.append(self.centerness(cls_tower_out)) 
            
            bbox_pred = self.scales[l](self.bbox_pred(reg_tower_out))
            bbox_pred = torch.exp(bbox_pred)
            bbox_reg.append(bbox_pred)
            
            centerness[-1] = self.centerness(reg_tower_out)

        return logits, bbox_reg, centerness
