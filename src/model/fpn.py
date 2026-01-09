import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    """
    Feature Pyramid Network.
    Takes [C3, C4, C5] and returns [P3, P4, P5, P6, P7].
    """
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: list of integers [C3_channels, C4_channels, C5_channels]
            out_channels: output channels for all P levels
        """
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # C3, C4, C5 lateral 1x1 convs
        for in_channels in in_channels_list:
            l_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.lateral_convs.append(l_conv)
            
            # 3x3 output convs for P3, P4, P5
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.fpn_convs.append(fpn_conv)
            
        # P6 and P7 generation layers 
        # (FCOS uses P6/P7 from P5 usually, or C5. Let's follow standard FCOS: P6 from P5, P7 from P6)
        self.p6_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        Args:
            inputs: [C3, C4, C5]
        """
        # Top-down pathway
        # C5 -> P5
        p5 = self.lateral_convs[2](inputs[2])
        
        # C4 -> P4
        c4_lat = self.lateral_convs[1](inputs[1])
        p4 = c4_lat + F.interpolate(p5, scale_factor=2, mode="nearest")
        
        # C3 -> P3
        c3_lat = self.lateral_convs[0](inputs[0])
        p3 = c3_lat + F.interpolate(p4, scale_factor=2, mode="nearest")
        
        # Smooth P3, P4, P5
        p3_out = self.fpn_convs[0](p3)
        p4_out = self.fpn_convs[1](p4)
        p5_out = self.fpn_convs[2](p5)
        
        # P6, P7
        p6_out = self.p6_conv(p5_out)
        p7_out = self.p7_conv(F.relu(p6_out)) # FCOS paper mentions Relu before P7 sometimes, let's stick to simple stride
        
        return [p3_out, p4_out, p5_out, p6_out, p7_out]
